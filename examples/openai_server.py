#!/usr/bin/env python3
"""
OpenAI-compatible TTS API server for faster-qwen3-tts.

Exposes POST /v1/audio/speech compatible with OpenAI's TTS API, enabling
integration with OpenWebUI, llama-swap, and other OpenAI-compatible clients.

Usage:
    pip install "faster-qwen3-tts[demo]"

    # Single default voice:
    python examples/openai_server.py \\
        --ref-audio voice.wav --ref-text "Reference transcription" \\
        --language English

    # Multiple named voices from a JSON config:
    python examples/openai_server.py --voices voices.json

    # Custom model and port:
    python examples/openai_server.py \\
        --model Qwen/Qwen3-TTS-12Hz-0.6B-Base \\
        --ref-audio voice.wav --ref-text "transcript" \\
        --port 8000

Voices config (voices.json):
    {
        "alloy": {"ref_audio": "voice.wav", "ref_text": "...", "language": "English"},
        "echo":  {"ref_audio": "voice2.wav", "ref_text": "...", "language": "English"}
    }

API usage:
    curl -s http://localhost:8000/v1/audio/speech \\
        -H "Content-Type: application/json" \\
        -d '{"model": "tts-1", "input": "Hello!", "voice": "alloy", "response_format": "wav"}' \\
        --output speech.wav
"""
import argparse
import asyncio
import base64
import hashlib
import io
import json
import logging
import os
import queue
import re
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import AsyncGenerator, Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

app = FastAPI(title="faster-qwen3-tts OpenAI-compatible API")

tts_model = None
voices: dict = {}
default_voice: Optional[str] = None
SAMPLE_RATE = 24000  # updated once the model loads
_model_lock = threading.Lock()  # prevent concurrent GPU inference

# SkyrimNet latency tuning.  Qwen3-TTS uses a 12 Hz codec, so 4 frames are
# roughly 333 ms of audio.  Keep these configurable for compatibility testing.
_XTTS_STREAMING = os.environ.get("QWEN_TTS_XTTS_STREAMING", "1").strip().lower() not in {
    "0", "false", "no", "off"
}
_XTTS_CHUNK_SIZE = max(1, int(os.environ.get("QWEN_TTS_XTTS_CHUNK_SIZE", "4")))
_STARTUP_WARMUP_PREFILL = max(
    1, int(os.environ.get("QWEN_TTS_WARMUP_PREFILL", "32"))
)

_skyrimnet_voice_dir = Path(
    os.environ.get(
        "QWEN_TTS_SKYRIMNET_VOICES_DIR",
        "/tmp/faster-qwen3-tts-skyrimnet-voices",
    )
)

_SKYRIMNET_LANGUAGE_MAP = {
    "auto": "Auto",
    "zh": "Chinese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "fr": "French",
    "ru": "Russian",
    "pt": "Portuguese",
    "es": "Spanish",
    "it": "Italian",
}

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class SpeechRequest(BaseModel):
    model: str = "tts-1"
    input: str
    voice: str = "alloy"
    response_format: str = "wav"  # wav | pcm | mp3
    speed: float = 1.0           # accepted but not yet applied
    language: Optional[str] = None      # override voice language for this request
    ref_audio: Optional[str] = None     # inline reference (path or base64); bypasses pool
    ref_text: Optional[str] = None      # inline reference transcript
    x_vector_only: Optional[bool] = None  # speaker-identity-only cloning; else voice/default
    parity_mode: Optional[bool] = None     # use upstream-style talker.generate path


class SkyrimNetSynthesisRequest(BaseModel):
    """Subset of the XTTS request schema used by SkyrimNet."""

    text: str = ""
    speaker_wav: Optional[str] = None
    language: Optional[str] = "en"
    accent: Optional[str] = None
    save_path: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    speed: Optional[float] = None
    repetition_penalty: Optional[float] = None
    override: Optional[bool] = False


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------


def _to_pcm16(pcm: np.ndarray) -> bytes:
    """Convert float32 numpy array to raw 16-bit little-endian PCM bytes."""
    return np.clip(pcm * 32768, -32768, 32767).astype(np.int16).tobytes()


def _wav_header(sample_rate: int, data_len: int = 0xFFFFFFFF) -> bytes:
    """Build a WAV header.  Use data_len=0xFFFFFFFF for streaming (unknown size)."""
    n_channels = 1
    bits = 16
    byte_rate = sample_rate * n_channels * bits // 8
    block_align = n_channels * bits // 8
    riff_size = 0xFFFFFFFF if data_len == 0xFFFFFFFF else 36 + data_len
    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", riff_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, n_channels, sample_rate,
                          byte_rate, block_align, bits))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_len))
    return buf.getvalue()


def _to_wav_bytes(pcm: np.ndarray, sample_rate: int) -> bytes:
    """Convert float32 numpy array to a complete WAV file in memory."""
    raw = _to_pcm16(pcm)
    return _wav_header(sample_rate, len(raw)) + raw


def _to_mp3_bytes(pcm: np.ndarray, sample_rate: int) -> bytes:
    """Convert float32 numpy array to MP3 bytes (requires pydub + ffmpeg)."""
    try:
        from pydub import AudioSegment
    except ImportError:
        raise HTTPException(
            status_code=400,
            detail="response_format='mp3' requires pydub: pip install pydub",
        )
    segment = AudioSegment(
        _to_pcm16(pcm),
        frame_rate=sample_rate,
        sample_width=2,
        channels=1,
    )
    buf = io.BytesIO()
    segment.export(buf, format="mp3")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Voice resolution
# ---------------------------------------------------------------------------


def _materialize_ref(value: str) -> str:
    """Return a filesystem path for inline ref_audio (a path, or base64 / data-URI)."""
    if os.path.isfile(value):
        return value
    m = re.match(r"data:[^,;]*;base64,(.*)$", value, re.DOTALL)
    raw = m.group(1) if m else value
    raw = re.sub(r"\s+", "", raw)
    try:
        data = base64.b64decode(raw, validate=True)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="inline ref_audio must be a filesystem path or base64 (data URI or raw)",
        )
    path = "/tmp/ref_" + hashlib.md5(value.encode()).hexdigest()[:16] + ".wav"
    if not os.path.exists(path):
        with open(path, "wb") as fh:
            fh.write(data)
    return path


def _coalesce_voice(voice_name: str, req: "SpeechRequest") -> tuple[dict, bool]:
    """Per-field coalescing: request value -> voice config -> server default.

    A request that supplies ref_audio is valid even if voice_name is unknown,
    so callers can drive an ad-hoc voice entirely over HTTP.
    """
    try:
        cfg = resolve_voice(voice_name)
    except HTTPException:
        if getattr(req, "ref_audio", None):
            cfg = {}
        else:
            raise
    eff = dict(cfg)
    if req.language is not None:
        value = req.language.strip()
        eff["language"] = _SKYRIMNET_LANGUAGE_MAP.get(value.lower(), value)
    elif eff.get("language"):
        value = str(eff["language"]).strip()
        eff["language"] = _SKYRIMNET_LANGUAGE_MAP.get(value.lower(), value)
    if req.ref_text is not None:
        eff["ref_text"] = req.ref_text
    xvec = bool(eff.get("x_vector_only", False))
    if req.x_vector_only is not None:
        xvec = bool(req.x_vector_only)
    if getattr(req, "ref_audio", None):
        eff["ref_audio"] = _materialize_ref(req.ref_audio)
    if not eff.get("ref_audio"):
        raise HTTPException(
            status_code=400,
            detail="no reference audio available: configure a voice or pass ref_audio",
        )
    return eff, xvec


def resolve_voice(voice_name: str) -> dict:
    """Return voice config dict or fall back to default, else raise 400."""
    if voice_name in voices:
        return voices[voice_name]
    if default_voice and default_voice in voices:
        logger.warning(
            "Voice %r not configured; falling back to default voice %r",
            voice_name,
            default_voice,
        )
        return voices[default_voice]
    raise HTTPException(
        status_code=400,
        detail=(
            f"Voice {voice_name!r} is not configured. "
            f"Available voices: {list(voices.keys())}"
        ),
    )


# ---------------------------------------------------------------------------
# SkyrimNet compatibility helpers
# ---------------------------------------------------------------------------


def _normalize_skyrimnet_language(language: Optional[str]) -> str:
    """Translate SkyrimNet/XTTS language codes to Qwen3-TTS language names."""
    if not language:
        return "Auto"
    value = language.strip()
    return _SKYRIMNET_LANGUAGE_MAP.get(value.lower(), value)


def _safe_runtime_voice_name(value: str) -> str:
    """Return a filesystem-safe runtime speaker name."""
    stem = Path(value).stem
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in stem)
    return safe.strip("._") or "speaker"


def _silence_wav_bytes(duration_ms: int = 100) -> bytes:
    """Generate a short mono PCM16 silence WAV used by SkyrimNet ping probes."""
    samples = max(1, int(SAMPLE_RATE * duration_ms / 1000))
    return _to_wav_bytes(np.zeros(samples, dtype=np.float32), SAMPLE_RATE)


def _resolve_skyrimnet_voice_name(speaker_wav: Optional[str]) -> Optional[str]:
    """Resolve SkyrimNet speaker identifiers against configured/runtime voices."""
    if speaker_wav:
        for candidate in (speaker_wav, Path(speaker_wav).stem):
            if candidate in voices:
                return candidate
    return default_voice


def _register_voice_aliases(speaker: str, cfg: dict, audio_path: Optional[Path] = None) -> None:
    """Register the aliases SkyrimNet may use later for the same NPC voice."""
    voices[speaker] = cfg
    voices[Path(speaker).stem] = cfg
    if audio_path is not None:
        voices[audio_path.name] = cfg
        voices[audio_path.stem] = cfg


def _normalize_skyrimnet_reference(raw: bytes, filename: str, output_path: Path) -> Path:
    """Decode an uploaded SkyrimNet reference into real mono PCM16 WAV.

    Skyrim voice assets can arrive through several conversion paths.  Speaker
    embeddings are much more stable when the encoder always receives the same
    straightforward waveform format, regardless of the source container/codec.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = Path(filename or "reference.wav").suffix or ".wav"
    source_path = output_path.with_name(output_path.stem + ".upload" + suffix)
    source_path.write_bytes(raw)

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(source_path),
                "-ac",
                "1",
                "-ar",
                str(SAMPLE_RATE),
                "-c:a",
                "pcm_s16le",
                str(output_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        logger.info(
            "Normalized SkyrimNet reference %s -> mono PCM16 %d Hz",
            filename or source_path.name,
            SAMPLE_RATE,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        # Preserve previous behaviour as a compatibility fallback.  The DGX
        # image ships ffmpeg, but a source checkout should still remain usable.
        logger.warning(
            "Could not normalize SkyrimNet reference %s with ffmpeg (%s); "
            "storing the uploaded bytes unchanged",
            filename or source_path.name,
            exc,
        )
        output_path.write_bytes(raw)
    finally:
        try:
            source_path.unlink()
        except OSError:
            pass

    return output_path


def _precompute_xvector_prompt(cfg: dict) -> None:
    """Extract and cache the speaker embedding before the first synthesis call."""
    if tts_model is None or not bool(cfg.get("x_vector_only", False)):
        return
    if cfg.get("_voice_clone_prompt") is not None:
        return

    ref_audio = cfg.get("ref_audio")
    if not ref_audio:
        return

    started = time.perf_counter()
    with _model_lock:
        prompt_items = tts_model.model.create_voice_clone_prompt(
            ref_audio=str(ref_audio),
            ref_text="",
            x_vector_only_mode=True,
        )
    spk_emb = prompt_items[0].ref_spk_embedding
    cfg["_voice_clone_prompt"] = {
        "ref_code": [None],
        "ref_spk_embedding": [spk_emb],
        "x_vector_only_mode": [True],
        "icl_mode": [False],
    }
    logger.info(
        "Precomputed x-vector prompt for %s in %.1f ms",
        ref_audio,
        (time.perf_counter() - started) * 1000.0,
    )


def _persist_skyrimnet_voice(speaker: str, cfg: dict, audio_path: Path) -> None:
    """Persist runtime voice metadata so container restarts do not lose registrations."""
    _skyrimnet_voice_dir.mkdir(parents=True, exist_ok=True)
    meta_path = _skyrimnet_voice_dir / (
        _safe_runtime_voice_name(speaker) + ".json"
    )
    payload = {
        "speaker_name": speaker,
        "audio_file": audio_path.name,
        "language": cfg.get("language", "Auto"),
        "ref_text": cfg.get("ref_text", ""),
        "x_vector_only": bool(cfg.get("x_vector_only", False)),
    }
    meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_skyrimnet_voices() -> int:
    """Restore persisted SkyrimNet runtime voices from sidecar metadata."""
    if not _skyrimnet_voice_dir.exists():
        return 0

    loaded = 0
    for meta_path in sorted(_skyrimnet_voice_dir.glob("*.json")):
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
            speaker = str(payload["speaker_name"])
            audio_path = _skyrimnet_voice_dir / str(payload["audio_file"])
            if not audio_path.is_file():
                logger.warning(
                    "Skipping SkyrimNet voice %s: missing audio file %s",
                    speaker,
                    audio_path,
                )
                continue
            cfg = {
                "ref_audio": str(audio_path),
                "ref_text": str(payload.get("ref_text") or ""),
                "language": str(payload.get("language") or "Auto"),
                "x_vector_only": bool(payload.get("x_vector_only", True)),
            }
            _register_voice_aliases(speaker, cfg, audio_path)
            loaded += 1
        except Exception as exc:
            logger.warning(
                "Skipping invalid SkyrimNet voice metadata %s: %s",
                meta_path,
                exc,
            )
    return loaded


async def _generate_complete_clone(
    req: SpeechRequest,
    generation_kwargs: Optional[dict] = None,
) -> tuple[np.ndarray, int]:
    """Generate a complete clone result through the shared Qwen request path."""
    if tts_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not req.input.strip():
        raise HTTPException(status_code=400, detail="'input' text is empty")

    voice_cfg, xvec = _coalesce_voice(req.voice, req)
    loop = asyncio.get_running_loop()
    generation_kwargs = dict(generation_kwargs or {})
    if req.parity_mode is not None:
        generation_kwargs["parity_mode"] = bool(req.parity_mode)

    def _generate():
        with _model_lock:
            return tts_model.generate_voice_clone(
                text=req.input,
                language=voice_cfg.get("language", "Auto"),
                ref_audio=voice_cfg["ref_audio"],
                ref_text=voice_cfg.get("ref_text", ""),
                xvec_only=xvec,
                voice_clone_prompt=voice_cfg.get("_voice_clone_prompt"),
                **generation_kwargs,
            )

    audio_arrays, sr = await loop.run_in_executor(None, _generate)
    audio = audio_arrays[0] if audio_arrays else np.zeros(1, dtype=np.float32)
    return audio, sr


async def _register_skyrimnet_voice(request: Request) -> dict:
    """Accept SkyrimNet's multipart voice upload and register it for Qwen cloning.

    SkyrimNet does not provide the reference transcript used by Qwen ICL cloning,
    so uploads default to x-vector-only cloning. A caller may optionally provide
    ref_text; when present, regular ICL cloning is used instead.
    """
    content_type = request.headers.get("content-type", "")
    fields: dict = {}
    upload = None

    if "application/json" in content_type:
        data = await request.json()
        if isinstance(data, dict):
            fields = data
    elif (
        "multipart/form-data" in content_type
        or "application/x-www-form-urlencoded" in content_type
    ):
        form = await request.form()
        for key, value in form.multi_items():
            if hasattr(value, "filename") and hasattr(value, "read"):
                if upload is None:
                    upload = value
                fields[key] = getattr(value, "filename", None)
            else:
                fields[key] = value

    speaker = (
        fields.get("speaker_name")
        or fields.get("speaker")
        or fields.get("speaker_wav")
    )
    if not speaker:
        raise HTTPException(status_code=400, detail="speaker_name is required")

    speaker = str(speaker)
    language = _normalize_skyrimnet_language(fields.get("language"))
    ref_text = str(fields.get("ref_text") or "")
    ref_audio = fields.get("ref_audio")

    saved_path: Optional[Path] = None
    if upload is not None:
        _skyrimnet_voice_dir.mkdir(parents=True, exist_ok=True)
        saved_path = _skyrimnet_voice_dir / (
            _safe_runtime_voice_name(speaker) + ".wav"
        )
        raw = await upload.read()
        if not raw:
            raise HTTPException(status_code=400, detail="wav_file is empty")
        _normalize_skyrimnet_reference(
            raw,
            getattr(upload, "filename", None) or saved_path.name,
            saved_path,
        )
        ref_audio = str(saved_path)
    elif ref_audio:
        ref_audio = _materialize_ref(str(ref_audio))
        saved_path = Path(ref_audio)

    if ref_audio:
        cfg = {
            "ref_audio": str(ref_audio),
            "ref_text": ref_text,
            "language": language,
            "x_vector_only": not bool(ref_text.strip()),
        }
        # SkyrimNet may later send either the speaker key or "speaker.wav".
        _register_voice_aliases(speaker, cfg, saved_path)
        if saved_path is not None and saved_path.parent == _skyrimnet_voice_dir:
            _persist_skyrimnet_voice(speaker, cfg, saved_path)

        # Move speaker-embedding extraction out of the first spoken line.
        # SkyrimNet already waits for enrollment, so paying this cost here
        # removes it from the lip-sync -> first-audio critical path.
        if cfg["x_vector_only"] and tts_model is not None:
            loop = asyncio.get_running_loop()
            try:
                await loop.run_in_executor(None, _precompute_xvector_prompt, cfg)
            except Exception:
                logger.exception(
                    "Failed to precompute x-vector for %s; synthesis will retry lazily",
                    speaker,
                )

        return {
            "success": True,
            "backend": "faster-qwen3-tts",
            "speaker_name": speaker,
            "language": language,
            "x_vector_only": cfg["x_vector_only"],
            "ref_audio": str(ref_audio),
        }

    # Some SkyrimNet versions probe this endpoint before a sample is attached.
    # Keep that flow non-fatal while making it explicit that no voice was added.
    return {
        "success": True,
        "backend": "faster-qwen3-tts",
        "speaker_name": speaker,
        "language": language,
        "registered": False,
        "message": "No voice sample supplied; request accepted as a compatibility no-op.",
    }


# ---------------------------------------------------------------------------
# Streaming helper: run sync generator in a background thread
# ---------------------------------------------------------------------------


async def _stream_chunks(
    voice_cfg: dict,
    text: str,
    xvec_only: bool = False,
    generation_kwargs: Optional[dict] = None,
) -> AsyncGenerator[bytes, None]:
    """Run Qwen streaming generation off-thread and yield PCM16 chunks ASAP."""
    q: queue.Queue = queue.Queue()
    _DONE = object()
    generation_kwargs = dict(generation_kwargs or {})
    chunk_size = int(generation_kwargs.pop("chunk_size", voice_cfg.get("chunk_size", _XTTS_CHUNK_SIZE)))
    non_streaming_mode = bool(generation_kwargs.pop("non_streaming_mode", False))
    voice_clone_prompt = voice_cfg.get("_voice_clone_prompt")

    def producer():
        started = time.perf_counter()
        first = True
        try:
            with _model_lock:
                for chunk, _sr, timing in tts_model.generate_voice_clone_streaming(
                    text=text,
                    language=voice_cfg.get("language", "Auto"),
                    ref_audio=voice_cfg["ref_audio"],
                    ref_text=voice_cfg.get("ref_text", ""),
                    chunk_size=chunk_size,
                    xvec_only=xvec_only,
                    non_streaming_mode=non_streaming_mode,
                    voice_clone_prompt=voice_clone_prompt,
                    **generation_kwargs,
                ):
                    if first:
                        ttfa_ms = (time.perf_counter() - started) * 1000.0
                        logger.info(
                            "TTS first audio chunk: %.1f ms "
                            "(prefill %.1f ms, first decode %.1f ms, chunk=%d frames)",
                            ttfa_ms,
                            float(timing.get("prefill_ms", 0.0)),
                            float(timing.get("decode_ms", 0.0)),
                            chunk_size,
                        )
                        first = False
                    q.put(chunk)
        except Exception as exc:
            q.put(exc)
        finally:
            q.put(_DONE)

    thread = threading.Thread(target=producer, daemon=True)
    thread.start()

    loop = asyncio.get_event_loop()
    while True:
        item = await loop.run_in_executor(None, q.get)
        if item is _DONE:
            break
        if isinstance(item, Exception):
            raise item
        yield _to_pcm16(item)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": tts_model is not None}


@app.get("/v1/audio/voices")
async def openai_list_voices():
    """List configured and runtime voices in an OpenAI-compatible shape."""
    seen = set()
    data = []
    for name, cfg in voices.items():
        if name in seen:
            continue
        seen.add(name)
        data.append({
            "name": name,
            "voice": name,
            "language": cfg.get("language", "Auto"),
            "has_ref_text": bool(str(cfg.get("ref_text") or "").strip()),
            "x_vector_only": bool(cfg.get("x_vector_only", False)),
        })
    return {"object": "list", "data": data, "voices": data}


@app.post("/v1/audio/voices")
async def openai_enroll_voice(request: Request):
    """Enroll a reusable cloned voice using the vLLM/Higgs OpenAI-compatible contract.

    Expected multipart fields:
      - audio_file: uploaded reference audio
      - consent: consent/enrollment identifier
      - name: reusable voice name
      - ref_text: optional reference transcript
      - speaker_description: optional metadata
    """
    content_type = request.headers.get("content-type", "")
    if "multipart/form-data" not in content_type:
        raise HTTPException(
            status_code=415,
            detail="/v1/audio/voices expects multipart/form-data",
        )

    form = await request.form()
    upload = (
        form.get("audio_file")
        or form.get("file")
        or form.get("wav_file")
        or form.get("audio")
    )
    if upload is None or not hasattr(upload, "read"):
        raise HTTPException(status_code=400, detail="audio_file is required")

    name = (
        form.get("name")
        or form.get("voice_name")
        or form.get("speaker_name")
        or form.get("voice")
    )
    if not name:
        raise HTTPException(status_code=400, detail="name is required")
    name = str(name)

    consent = str(form.get("consent") or form.get("consent_id") or "skyrimnet")
    ref_text = str(form.get("ref_text") or form.get("transcript") or "").strip()
    speaker_description = str(form.get("speaker_description") or "").strip()

    _skyrimnet_voice_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(getattr(upload, "filename", "") or "").suffix.lower()
    if not suffix:
        suffix = ".wav"
    saved_path = _skyrimnet_voice_dir / (
        _safe_runtime_voice_name(name) + suffix
    )
    raw = await upload.read()
    if not raw:
        raise HTTPException(status_code=400, detail="audio_file is empty")
    saved_path.write_bytes(raw)

    language = _normalize_skyrimnet_language(
        str(form.get("language") or "Auto")
    )
    cfg = {
        "ref_audio": str(saved_path),
        "ref_text": ref_text,
        "language": language,
        "x_vector_only": not bool(ref_text),
    }
    _register_voice_aliases(name, cfg, saved_path)
    _persist_skyrimnet_voice(name, cfg, saved_path)

    import time
    result = {
        "name": name,
        "voice": name,
        "id": name,
        "consent": consent,
        "created_at": int(time.time()),
        "mime_type": getattr(upload, "content_type", None) or "application/octet-stream",
        "file_size": len(raw),
        "language": language,
        "ref_text": ref_text,
        "x_vector_only": cfg["x_vector_only"],
    }
    if speaker_description:
        result["speaker_description"] = speaker_description

    logger.info(
        "Enrolled OpenAI-compatible voice %r (%d bytes, ref_text=%s, xvec=%s)",
        name,
        len(raw),
        bool(ref_text),
        cfg["x_vector_only"],
    )
    return result


@app.post("/create_and_store_latents")
@app.post("/create_and_store_latents/")
async def skyrimnet_create_and_store_latents(request: Request):
    """Register a SkyrimNet NPC voice sample for subsequent synthesis."""
    return await _register_skyrimnet_voice(request)


@app.post("/tts_to_audio")
@app.post("/tts_to_audio/")
async def skyrimnet_tts_to_audio(req: SkyrimNetSynthesisRequest):
    """XTTS-compatible synthesis endpoint used by SkyrimNet."""
    if tts_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    text = req.text.strip()
    if not text or text == "ping":
        headers = {}
        if req.save_path:
            headers["Content-Disposition"] = (
                f'attachment; filename="{Path(req.save_path).name.replace(chr(34), "_")}"'
            )
        return Response(
            content=_silence_wav_bytes(),
            media_type="audio/wav",
            headers=headers,
        )

    voice_name = _resolve_skyrimnet_voice_name(req.speaker_wav)
    if not voice_name:
        raise HTTPException(
            status_code=400,
            detail=(
                "No SkyrimNet voice is registered. Upload one through "
                "/create_and_store_latents or configure --ref-audio/--voices."
            ),
        )

    qreq = SpeechRequest(
        input=req.text,
        voice=voice_name,
        response_format="wav",
        language=_normalize_skyrimnet_language(req.language),
        x_vector_only=True,
    )
    generation_kwargs = {
        # Critical for low TTFT: do NOT prefill the whole target sentence.
        # Qwen can feed target text step-by-step while acoustic decode is already
        # running.  This is also the mode that produced the clean Veronica x-vector
        # result in our cross-language tests.
        "non_streaming_mode": False,
        "chunk_size": _XTTS_CHUNK_SIZE,
    }
    if req.temperature is not None:
        generation_kwargs["temperature"] = req.temperature
    if req.top_p is not None:
        generation_kwargs["top_p"] = req.top_p
    if req.top_k is not None:
        generation_kwargs["top_k"] = req.top_k
    if req.repetition_penalty is not None:
        generation_kwargs["repetition_penalty"] = req.repetition_penalty

    headers = {}
    if req.save_path:
        headers["Content-Disposition"] = (
            f'attachment; filename="{Path(req.save_path).name.replace(chr(34), "_")}"'
        )

    if _XTTS_STREAMING:
        voice_cfg, xvec = _coalesce_voice(voice_name, qreq)

        async def xtts_audio_stream():
            # Send the WAV header immediately, then audio as Qwen emits codec chunks.
            yield _wav_header(SAMPLE_RATE)
            async for raw_chunk in _stream_chunks(
                voice_cfg,
                qreq.input,
                xvec,
                generation_kwargs,
            ):
                yield raw_chunk

        return StreamingResponse(
            xtts_audio_stream(),
            media_type="audio/wav",
            headers=headers,
        )

    # Compatibility escape hatch for clients that require a complete WAV body.
    complete_kwargs = dict(generation_kwargs)
    complete_kwargs.pop("chunk_size", None)
    audio, sr = await _generate_complete_clone(qreq, complete_kwargs)
    return Response(
        content=_to_wav_bytes(audio, sr),
        media_type="audio/wav",
        headers=headers,
    )


@app.post("/v1/audio/speech")
async def create_speech(req: SpeechRequest):
    if tts_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not req.input.strip():
        raise HTTPException(status_code=400, detail="'input' text is empty")

    voice_cfg, _xvec = _coalesce_voice(req.voice, req)
    fmt = req.response_format.lower()

    _CONTENT_TYPES = {
        "wav": "audio/wav",
        "pcm": "audio/pcm",
        "mp3": "audio/mpeg",
    }
    if fmt not in _CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"response_format {fmt!r} not supported. Use: wav, pcm, mp3",
        )
    content_type = _CONTENT_TYPES[fmt]

    # --- MP3: generate all audio, then encode (non-streaming) ---
    if fmt == "mp3":
        audio, sr = await _generate_complete_clone(req)
        return Response(content=_to_mp3_bytes(audio, sr), media_type=content_type)

    # --- WAV / PCM: stream chunks as they are generated ---
    async def audio_stream():
        if fmt == "wav":
            yield _wav_header(SAMPLE_RATE)  # stream with unknown data length
        async for raw_chunk in _stream_chunks(voice_cfg, req.input, _xvec):
            yield raw_chunk

    return StreamingResponse(audio_stream(), media_type=content_type)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _parse_args():
    p = argparse.ArgumentParser(
        description="OpenAI-compatible TTS server for faster-qwen3-tts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--model",
        default=os.environ.get("QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"),
        help="HuggingFace model ID or local path (default: Qwen/Qwen3-TTS-12Hz-1.7B-Base)",
    )
    p.add_argument(
        "--voices",
        default=os.environ.get("QWEN_TTS_VOICES"),
        metavar="FILE",
        help="JSON file mapping voice names to {ref_audio, ref_text, language}",
    )
    p.add_argument(
        "--ref-audio",
        default=os.environ.get("QWEN_TTS_REF_AUDIO"),
        metavar="FILE",
        help="Reference audio file when --voices is not used",
    )
    p.add_argument(
        "--ref-text",
        default=os.environ.get("QWEN_TTS_REF_TEXT", ""),
        help="Transcript of --ref-audio",
    )
    p.add_argument(
        "--language",
        default=os.environ.get("QWEN_TTS_LANGUAGE", "Auto"),
        help="Target language (English, French, Auto, …) when --voices is not used",
    )
    p.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    p.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    p.add_argument("--device", default="cuda", help="Torch device (default: cuda)")
    return p.parse_args()


def main():
    global tts_model, voices, default_voice, SAMPLE_RATE

    args = _parse_args()

    # Build voice registry
    if args.voices:
        with open(args.voices) as f:
            voices = json.load(f)
        default_voice = next(iter(voices))
        logger.info("Loaded %d voice(s) from %s", len(voices), args.voices)
    elif args.ref_audio:
        voices = {
            "default": {
                "ref_audio": args.ref_audio,
                "ref_text": args.ref_text,
                "language": args.language,
            }
        }
        default_voice = "default"
        logger.info("Using single voice from --ref-audio: %s", args.ref_audio)
    else:
        # Valid for per-request OpenAI ref_audio and SkyrimNet runtime uploads.
        voices = {}
        default_voice = None
        logger.info(
            "Starting without a preconfigured voice; provide ref_audio per request "
            "or register SkyrimNet voices at /create_and_store_latents."
        )

    restored = _load_skyrimnet_voices()
    if restored:
        logger.info(
            "Restored %d persisted SkyrimNet runtime voice(s) from %s",
            restored,
            _skyrimnet_voice_dir,
        )

    from faster_qwen3_tts import FasterQwen3TTS

    logger.info("Loading model %s on %s …", args.model, args.device)
    tts_model = FasterQwen3TTS.from_pretrained(
        args.model,
        device=args.device,
        dtype=torch.bfloat16,
    )
    SAMPLE_RATE = tts_model.sample_rate

    # Capture CUDA graphs before accepting requests so the first NPC line does
    # not pay graph-capture latency while its lip-sync is already running.
    warm_started = time.perf_counter()
    tts_model.warmup(prefill_len=_STARTUP_WARMUP_PREFILL)
    logger.info(
        "CUDA graph warmup completed in %.1f ms (prefill=%d)",
        (time.perf_counter() - warm_started) * 1000.0,
        _STARTUP_WARMUP_PREFILL,
    )

    logger.info("Model ready. Sample rate: %d Hz", SAMPLE_RATE)
    logger.info(
        "SkyrimNet XTTS streaming=%s chunk_size=%d",
        _XTTS_STREAMING,
        _XTTS_CHUNK_SIZE,
    )
    logger.info("Server listening on http://%s:%d", args.host, args.port)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
