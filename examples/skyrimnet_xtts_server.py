#!/usr/bin/env python3
"""
SkyrimNet XTTS-compatible server for faster-qwen3-tts.

Only the endpoints SkyrimNet's XTTS provider needs are exposed:
  GET  /health
  POST /create_and_store_latents[/]
  POST /tts_to_audio[/]

This intentionally does not expose the OpenAI TTS API.
"""
import argparse
import asyncio
import inspect
import io
import json
import logging
import os
import queue
import struct
import subprocess
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

from faster_qwen3_tts import FasterQwen3TTS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("skyrimnet_xtts")

app = FastAPI(title="SkyrimNet Qwen3-TTS XTTS shim")

tts_model = None
voices: dict = {}
default_voice: Optional[str] = None
SAMPLE_RATE = 24000
_model_lock = threading.Lock()

_RUNTIME_DIR = Path(
    os.environ.get(
        "QWEN_TTS_SKYRIMNET_VOICES_DIR",
        "/tmp/faster-qwen3-tts-skyrimnet-voices",
    )
)
_HTTP_STREAMING = os.environ.get("QWEN_TTS_XTTS_STREAMING", "1").lower() not in {
    "0", "false", "no", "off"
}
_STREAM_CHUNK_FRAMES = max(
    1, int(os.environ.get("QWEN_TTS_XTTS_CHUNK_SIZE", "4"))
)
_WARMUP_PREFILL = max(
    1, int(os.environ.get("QWEN_TTS_WARMUP_PREFILL", "32"))
)

_LANGUAGE_MAP = {
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


class SynthesisRequest(BaseModel):
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


def _normalize_language(value: Optional[str]) -> str:
    if not value:
        return "Auto"
    value = str(value).strip()
    return _LANGUAGE_MAP.get(value.lower(), value)


def _safe_name(value: str) -> str:
    stem = Path(value).stem
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in stem)
    return safe.strip("._") or "speaker"


def _to_pcm16(audio: np.ndarray) -> bytes:
    return np.clip(audio * 32768.0, -32768, 32767).astype(np.int16).tobytes()


def _wav_header(sample_rate: int, data_len: int = 0xFFFFFFFF) -> bytes:
    byte_rate = sample_rate * 2
    riff_size = 0xFFFFFFFF if data_len == 0xFFFFFFFF else 36 + data_len
    return (
        b"RIFF"
        + struct.pack("<I", riff_size)
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, byte_rate, 2, 16)
        + b"data"
        + struct.pack("<I", data_len)
    )


def _to_wav(audio: np.ndarray, sample_rate: int) -> bytes:
    raw = _to_pcm16(audio)
    return _wav_header(sample_rate, len(raw)) + raw


def _silence_wav(duration_ms: int = 100) -> bytes:
    n = max(1, int(SAMPLE_RATE * duration_ms / 1000))
    return _to_wav(np.zeros(n, dtype=np.float32), SAMPLE_RATE)


def _accepts(method, name: str) -> bool:
    try:
        sig = inspect.signature(method)
    except (TypeError, ValueError):
        return False
    return name in sig.parameters or any(
        p.kind is inspect.Parameter.VAR_KEYWORD
        for p in sig.parameters.values()
    )


def _warm_model() -> None:
    method = getattr(tts_model, "warmup", None)
    if method is None:
        method = getattr(tts_model, "_warmup", None)
    if method is None:
        logger.warning("No explicit warmup API; first request will capture graphs lazily")
        return

    started = time.perf_counter()
    try:
        method(prefill_len=_WARMUP_PREFILL)
    except TypeError:
        method(_WARMUP_PREFILL)
    logger.info(
        "CUDA graph warmup complete in %.1f ms (prefill=%d)",
        (time.perf_counter() - started) * 1000.0,
        _WARMUP_PREFILL,
    )


def _normalize_uploaded_reference(raw: bytes, filename: str, dst: Path) -> None:
    """Always give the speaker encoder a real PCM16 mono 24 kHz WAV."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    suffix = Path(filename or "voice.wav").suffix or ".bin"
    src = dst.with_name(dst.stem + ".upload" + suffix)
    src.write_bytes(raw)

    try:
        proc = subprocess.run(
            [
                "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(src),
                "-ac", "1",
                "-ar", str(SAMPLE_RATE),
                "-c:a", "pcm_s16le",
                str(dst),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=False,
        )
        if proc.returncode != 0:
            detail = proc.stderr.decode("utf-8", "replace")[-1000:]
            raise HTTPException(
                status_code=400,
                detail=f"Could not decode voice reference with ffmpeg: {detail}",
            )
        logger.info(
            "Normalized reference %s -> PCM16 mono %d Hz",
            filename,
            SAMPLE_RATE,
        )
    finally:
        try:
            src.unlink()
        except OSError:
            pass


def _register_aliases(name: str, cfg: dict, audio_path: Optional[Path]) -> None:
    voices[name] = cfg
    voices[Path(name).stem] = cfg
    if audio_path is not None:
        voices[audio_path.name] = cfg
        voices[audio_path.stem] = cfg


def _persist_voice(name: str, cfg: dict, audio_path: Path) -> None:
    _RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    meta = {
        "speaker_name": name,
        "audio_file": audio_path.name,
        "language": cfg.get("language", "Auto"),
        "ref_text": cfg.get("ref_text", ""),
        "x_vector_only": True,
    }
    (_RUNTIME_DIR / (_safe_name(name) + ".json")).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _load_runtime_voices() -> int:
    if not _RUNTIME_DIR.exists():
        return 0
    loaded = 0
    for meta_path in sorted(_RUNTIME_DIR.glob("*.json")):
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
            name = str(data["speaker_name"])
            audio = _RUNTIME_DIR / str(data["audio_file"])
            if not audio.is_file():
                continue
            cfg = {
                "ref_audio": str(audio),
                "ref_text": "",
                "language": str(data.get("language") or "Auto"),
                "x_vector_only": True,
                "_runtime": True,
            }
            _register_aliases(name, cfg, audio)
            loaded += 1
        except Exception:
            logger.exception("Could not restore runtime voice from %s", meta_path)
    return loaded


def _precompute_xvector(cfg: dict) -> None:
    """Pay speaker-encoder cost before the NPC starts a spoken line."""
    if tts_model is None or cfg.get("_xvector_ready"):
        return

    ref_audio = str(cfg["ref_audio"])
    started = time.perf_counter()

    with _model_lock:
        prompt_items = tts_model.model.create_voice_clone_prompt(
            ref_audio=ref_audio,
            ref_text="",
            x_vector_only_mode=True,
        )

    spk_emb = prompt_items[0].ref_spk_embedding
    vcp = {
        "ref_code": [None],
        "ref_spk_embedding": [spk_emb],
        "x_vector_only_mode": [True],
        "icl_mode": [False],
    }
    cfg["_voice_clone_prompt"] = vcp
    cfg["_xvector_ready"] = True

    # Older releases use this private cache instead of a voice_clone_prompt arg.
    cache = getattr(tts_model, "_voice_prompt_cache", None)
    if isinstance(cache, dict):
        # x-vector mode ignores append_silence, but older releases key it anyway.
        cache[(ref_audio, "", True, True)] = (vcp, [None])
        cache[(ref_audio, "", True, False)] = (vcp, [None])

    logger.info(
        "Prepared x-vector for %s in %.1f ms",
        ref_audio,
        (time.perf_counter() - started) * 1000.0,
    )


def _resolve_voice(speaker_wav: Optional[str]) -> tuple[str, dict]:
    if speaker_wav:
        for candidate in (speaker_wav, Path(speaker_wav).stem):
            if candidate in voices:
                return candidate, voices[candidate]
    if default_voice and default_voice in voices:
        return default_voice, voices[default_voice]
    raise HTTPException(status_code=400, detail="No registered SkyrimNet voice")


def _generation_kwargs(method, cfg: dict, req: SynthesisRequest) -> dict:
    kwargs = {
        "text": req.text,
        "language": _normalize_language(req.language),
        "ref_audio": cfg["ref_audio"],
        "ref_text": "",
        "xvec_only": True,
    }

    # This is the critical low-TTFT mode: target text is fed progressively
    # instead of being entirely included in the acoustic prefill.
    if _accepts(method, "non_streaming_mode"):
        kwargs["non_streaming_mode"] = False

    prompt = cfg.get("_voice_clone_prompt")
    if prompt is not None and _accepts(method, "voice_clone_prompt"):
        kwargs["voice_clone_prompt"] = prompt

    optional = {
        "temperature": req.temperature,
        "top_p": req.top_p,
        "top_k": req.top_k,
        "repetition_penalty": req.repetition_penalty,
    }
    for key, value in optional.items():
        if value is not None and _accepts(method, key):
            kwargs[key] = value

    return kwargs


async def _stream_audio(cfg: dict, req: SynthesisRequest) -> AsyncGenerator[bytes, None]:
    method = getattr(tts_model, "generate_voice_clone_streaming", None)
    if method is None:
        raise RuntimeError("Installed faster-qwen3-tts has no streaming generator")

    q: queue.Queue = queue.Queue()
    DONE = object()

    def producer():
        started = time.perf_counter()
        first = True
        try:
            kwargs = _generation_kwargs(method, cfg, req)
            if _accepts(method, "chunk_size"):
                kwargs["chunk_size"] = _STREAM_CHUNK_FRAMES

            with _model_lock:
                for item in method(**kwargs):
                    if not isinstance(item, tuple):
                        raise RuntimeError("Unexpected streaming generator result")
                    if len(item) == 3:
                        chunk, _sr, timing = item
                    elif len(item) == 2:
                        chunk, _sr = item
                        timing = {}
                    else:
                        raise RuntimeError(
                            f"Unexpected streaming generator tuple length: {len(item)}"
                        )

                    if first:
                        logger.info(
                            "TTFA %.1f ms | prefill %.1f ms | first-decode %.1f ms | chunk=%d",
                            (time.perf_counter() - started) * 1000.0,
                            float(timing.get("prefill_ms", 0.0)),
                            float(timing.get("decode_ms", 0.0)),
                            _STREAM_CHUNK_FRAMES,
                        )
                        first = False
                    q.put(chunk)
        except Exception as exc:
            q.put(exc)
        finally:
            q.put(DONE)

    threading.Thread(target=producer, daemon=True).start()
    loop = asyncio.get_running_loop()
    while True:
        item = await loop.run_in_executor(None, q.get)
        if item is DONE:
            return
        if isinstance(item, Exception):
            raise item
        yield _to_pcm16(item)


async def _generate_complete(cfg: dict, req: SynthesisRequest) -> tuple[np.ndarray, int]:
    method = tts_model.generate_voice_clone
    kwargs = _generation_kwargs(method, cfg, req)
    loop = asyncio.get_running_loop()

    def run():
        with _model_lock:
            return method(**kwargs)

    audio_arrays, sr = await loop.run_in_executor(None, run)
    audio = audio_arrays[0] if audio_arrays else np.zeros(1, dtype=np.float32)
    return audio, sr


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": tts_model is not None,
        "provider": "skyrimnet-xtts",
        "http_streaming": _HTTP_STREAMING,
        "chunk_frames": _STREAM_CHUNK_FRAMES,
    }


@app.post("/create_and_store_latents")
@app.post("/create_and_store_latents/")
async def create_and_store_latents(request: Request):
    form = await request.form()

    speaker = (
        form.get("speaker_name")
        or form.get("speaker")
        or form.get("speaker_wav")
    )
    if not speaker:
        raise HTTPException(status_code=400, detail="speaker_name is required")
    speaker = str(speaker)

    upload = (
        form.get("wav_file")
        or form.get("audio_file")
        or form.get("file")
        or form.get("audio")
    )
    if upload is None or not hasattr(upload, "read"):
        raise HTTPException(status_code=400, detail="wav_file is required")

    raw = await upload.read()
    if not raw:
        raise HTTPException(status_code=400, detail="wav_file is empty")

    dst = _RUNTIME_DIR / (_safe_name(speaker) + ".wav")
    _normalize_uploaded_reference(
        raw,
        getattr(upload, "filename", None) or dst.name,
        dst,
    )

    cfg = {
        "ref_audio": str(dst),
        "ref_text": "",
        "language": _normalize_language(form.get("language")),
        "x_vector_only": True,
        "_runtime": True,
    }
    _register_aliases(speaker, cfg, dst)
    _persist_voice(speaker, cfg, dst)

    # Enrollment is the right place to pay the speaker-encoder cost.
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _precompute_xvector, cfg)

    return {
        "success": True,
        "speaker_name": speaker,
        "language": cfg["language"],
        "x_vector_only": True,
        "normalized": "pcm_s16le/mono/24000",
    }


@app.post("/tts_to_audio")
@app.post("/tts_to_audio/")
async def tts_to_audio(req: SynthesisRequest):
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
            content=_silence_wav(),
            media_type="audio/wav",
            headers=headers,
        )

    _name, cfg = _resolve_voice(req.speaker_wav)

    # Old persisted voices may not have been warmed yet if startup precompute failed.
    if not cfg.get("_xvector_ready"):
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _precompute_xvector, cfg)

    headers = {}
    if req.save_path:
        headers["Content-Disposition"] = (
            f'attachment; filename="{Path(req.save_path).name.replace(chr(34), "_")}"'
        )

    if _HTTP_STREAMING and hasattr(tts_model, "generate_voice_clone_streaming"):
        async def body():
            # Headers reach SkyrimNet immediately; PCM follows as soon as Qwen has
            # decoded the first audio chunk.
            yield _wav_header(SAMPLE_RATE)
            async for pcm in _stream_audio(cfg, req):
                yield pcm

        return StreamingResponse(body(), media_type="audio/wav", headers=headers)

    audio, sr = await _generate_complete(cfg, req)
    return Response(content=_to_wav(audio, sr), media_type="audio/wav", headers=headers)


def _parse_args():
    p = argparse.ArgumentParser(description="SkyrimNet XTTS shim for Qwen3-TTS")
    p.add_argument("--model", required=True)
    p.add_argument("--voices")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    global tts_model, voices, default_voice, SAMPLE_RATE

    args = _parse_args()

    if args.voices:
        with open(args.voices, encoding="utf-8") as fh:
            configured = json.load(fh)
        for name, cfg_in in configured.items():
            cfg = dict(cfg_in)
            cfg["language"] = _normalize_language(cfg.get("language"))
            # SkyrimNet path is identity-only by design.
            cfg["x_vector_only"] = bool(cfg.get("x_vector_only", cfg.get("xvector", False)))
            voices[name] = cfg
        default_voice = next(iter(configured), None)
        logger.info("Loaded %d configured voice(s)", len(configured))

    restored = _load_runtime_voices()
    logger.info("Restored %d persisted SkyrimNet voice(s)", restored)

    logger.info("Loading model %s on %s", args.model, args.device)
    tts_model = FasterQwen3TTS.from_pretrained(
        args.model,
        device=args.device,
        dtype=torch.bfloat16,
    )
    SAMPLE_RATE = int(getattr(tts_model, "sample_rate", 24000) or 24000)

    _warm_model()

    # Warm each unique x-vector voice before accepting traffic.
    seen = set()
    prepared = 0
    for cfg in voices.values():
        ident = id(cfg)
        if ident in seen:
            continue
        seen.add(ident)
        if not bool(cfg.get("x_vector_only", False)):
            continue
        try:
            _precompute_xvector(cfg)
            prepared += 1
        except Exception:
            logger.exception("Could not precompute %s", cfg.get("ref_audio"))
    logger.info("Prepared %d x-vector voice(s) before startup", prepared)

    logger.info(
        "Ready: sample_rate=%d, XTTS streaming=%s, chunk_frames=%d",
        SAMPLE_RATE,
        _HTTP_STREAMING,
        _STREAM_CHUNK_FRAMES,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
