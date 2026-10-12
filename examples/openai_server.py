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
import sys
import threading
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
        eff["language"] = req.language
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
        saved_path.write_bytes(await upload.read())
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
        voices[speaker] = cfg
        voices[Path(speaker).stem] = cfg
        if saved_path is not None:
            voices[saved_path.name] = cfg
            voices[saved_path.stem] = cfg

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


async def _stream_chunks(voice_cfg: dict, text: str, xvec_only: bool = False) -> AsyncGenerator[bytes, None]:
    """
    Run generate_voice_clone_streaming in a background thread and yield
    raw PCM bytes for each chunk as they arrive.
    """
    q: queue.Queue = queue.Queue()
    _DONE = object()

    def producer():
        try:
            with _model_lock:
                for chunk, _sr, _timing in tts_model.generate_voice_clone_streaming(
                    text=text,
                    language=voice_cfg.get("language", "Auto"),
                    ref_audio=voice_cfg["ref_audio"],
                    ref_text=voice_cfg.get("ref_text", ""),
                    chunk_size=voice_cfg.get("chunk_size", 12),
                    xvec_only=xvec_only,
                    non_streaming_mode=False,
                ):
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
    )
    voice_cfg, xvec = _coalesce_voice(qreq.voice, qreq)

    loop = asyncio.get_running_loop()

    def _generate():
        with _model_lock:
            return tts_model.generate_voice_clone(
                text=qreq.input,
                language=voice_cfg.get("language", "Auto"),
                ref_audio=voice_cfg["ref_audio"],
                ref_text=voice_cfg.get("ref_text", ""),
                xvec_only=xvec,
            )

    audio_arrays, sr = await loop.run_in_executor(None, _generate)
    audio = audio_arrays[0] if audio_arrays else np.zeros(1, dtype=np.float32)

    headers = {}
    if req.save_path:
        headers["Content-Disposition"] = (
            f'attachment; filename="{Path(req.save_path).name.replace(chr(34), "_")}"'
        )
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
        loop = asyncio.get_event_loop()

        def _generate():
            with _model_lock:
                return tts_model.generate_voice_clone(
                    text=req.input,
                    language=voice_cfg.get("language", "Auto"),
                    ref_audio=voice_cfg["ref_audio"],
                    ref_text=voice_cfg.get("ref_text", ""),
                    xvec_only=_xvec,
                )

        audio_arrays, sr = await loop.run_in_executor(None, _generate)
        audio = audio_arrays[0] if audio_arrays else np.zeros(1, dtype=np.float32)
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

    from faster_qwen3_tts import FasterQwen3TTS

    logger.info("Loading model %s on %s …", args.model, args.device)
    tts_model = FasterQwen3TTS.from_pretrained(
        args.model,
        device=args.device,
        dtype=torch.bfloat16,
    )
    SAMPLE_RATE = tts_model.sample_rate
    logger.info("Model ready. Sample rate: %d Hz", SAMPLE_RATE)
    logger.info("Server listening on http://%s:%d", args.host, args.port)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
