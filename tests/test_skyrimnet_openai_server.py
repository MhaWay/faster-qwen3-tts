"""Tests for the SkyrimNet compatibility surface in examples/openai_server.py."""

import importlib.util
import io
import wave
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")
pytest.importorskip("multipart")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient


MODULE_PATH = Path(__file__).resolve().parents[1] / "examples" / "openai_server.py"
SPEC = importlib.util.spec_from_file_location("faster_qwen3_tts_openai_server", MODULE_PATH)
server = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(server)


class DummyTTS:
    sample_rate = 24000

    def __init__(self):
        self.last_call = None

    def generate_voice_clone(self, **kwargs):
        self.last_call = kwargs
        return [np.zeros(240, dtype=np.float32)], self.sample_rate


@pytest.fixture(autouse=True)
def _reset_server_state(tmp_path, monkeypatch):
    dummy = DummyTTS()
    monkeypatch.setattr(server, "tts_model", dummy)
    monkeypatch.setattr(server, "SAMPLE_RATE", 24000)
    monkeypatch.setattr(server, "voices", {})
    monkeypatch.setattr(server, "default_voice", None)
    monkeypatch.setattr(server, "_skyrimnet_voice_dir", tmp_path / "voices")
    yield dummy


@pytest.fixture
def client():
    return TestClient(server.app)


def test_skyrimnet_ping_returns_valid_silence_wav(client):
    response = client.post(
        "/tts_to_audio/",
        json={"text": "ping", "speaker_wav": "unused", "language": "en"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/wav")
    with wave.open(io.BytesIO(response.content), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == 24000
        assert wav.getnframes() > 0


def test_skyrimnet_upload_registers_xvector_voice_and_synthesizes(client, _reset_server_state):
    response = client.post(
        "/create_and_store_latents/",
        data={"speaker_name": "ulfric", "language": "it"},
        files={"wav_file": ("ulfric.wav", b"RIFF-test-reference", "audio/wav")},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["success"] is True
    assert body["language"] == "Italian"
    assert body["x_vector_only"] is True
    assert "ulfric" in server.voices
    assert Path(server.voices["ulfric"]["ref_audio"]).exists()

    speech = client.post(
        "/tts_to_audio/",
        json={
            "text": "Fermati, viandante.",
            "speaker_wav": "ulfric",
            "language": "it",
            "save_path": "line.wav",
        },
    )

    assert speech.status_code == 200, speech.text
    assert speech.content.startswith(b"RIFF")
    assert 'filename="line.wav"' in speech.headers["content-disposition"]

    call = _reset_server_state.last_call
    assert call is not None
    assert call["language"] == "Italian"
    assert call["xvec_only"] is True
    assert call["ref_text"] == ""


def test_skyrimnet_optional_ref_text_enables_icl_mode(client):
    response = client.post(
        "/create_and_store_latents/",
        data={
            "speaker_name": "lydia",
            "language": "en",
            "ref_text": "I am sworn to carry your burdens.",
        },
        files={"wav_file": ("lydia.wav", b"RIFF-test-reference", "audio/wav")},
    )

    assert response.status_code == 200, response.text
    assert response.json()["x_vector_only"] is False
    assert server.voices["lydia"]["ref_text"] == "I am sworn to carry your burdens."


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("en", "English"),
        ("it", "Italian"),
        ("ja", "Japanese"),
        ("Auto", "Auto"),
        ("English", "English"),
    ],
)
def test_skyrimnet_language_mapping(source, expected):
    assert server._normalize_skyrimnet_language(source) == expected
