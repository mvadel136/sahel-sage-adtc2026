"""LlamaServerBackend's HTTP shape, with urlopen faked out.

No subprocess is ever spawned here and no socket is opened: only the request
the backend *would* send is inspected.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sahel_sage.app.backend import DEFAULT_STOP, LlamaServerBackend, LlamaServerConfig


class FakeResponse:
    def __init__(self, body: bytes = b"", lines: list[bytes] | None = None):
        self._body = body
        self._lines = lines or []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self) -> bytes:
        return self._body

    def __iter__(self):
        return iter(self._lines)


@pytest.fixture
def backend() -> LlamaServerBackend:
    cfg = LlamaServerConfig(model=Path("/nowhere/model.gguf"), binary=Path("/nowhere/llama-server"))
    b = LlamaServerBackend(cfg)
    b.port = 9999
    return b


def _capture(monkeypatch, response: FakeResponse) -> list:
    seen = []

    def fake_urlopen(req, timeout=None):
        seen.append(req)
        return response

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    return seen


def test_complete_uses_raw_completions_not_chat(backend, monkeypatch):
    """The shipped model is Base-trained on raw text; /v1/chat/completions would
    wrap the prompt in a template it has never seen."""
    body = json.dumps({"choices": [{"text": "LIKELY ISSUE: mildew"}]}).encode()
    seen = _capture(monkeypatch, FakeResponse(body=body))

    out = backend.complete("PROMPT\n\nSAHEL SAGE:\n", max_tokens=64, temperature=0.1)

    assert out == "LIKELY ISSUE: mildew"
    req = seen[0]
    assert req.full_url == "http://127.0.0.1:9999/v1/completions"
    payload = json.loads(req.data)
    assert payload["prompt"] == "PROMPT\n\nSAHEL SAGE:\n"
    assert "messages" not in payload
    assert payload["max_tokens"] == 64 and payload["temperature"] == 0.1
    assert payload["stop"] == list(DEFAULT_STOP)


def test_stream_yields_text_deltas_and_ignores_noise(backend, monkeypatch):
    lines = [
        b': keep-alive\n',
        b'data: {"choices":[{"text":"LIKELY "}]}\n',
        b'data: not json\n',
        b'data: {"choices":[{"text":"ISSUE"}]}\n',
        b'data: [DONE]\n',
        b'data: {"choices":[{"text":"never reached"}]}\n',
    ]
    seen = _capture(monkeypatch, FakeResponse(lines=lines))

    assert list(backend.stream("PROMPT")) == ["LIKELY ", "ISSUE"]
    assert json.loads(seen[0].data)["stream"] is True


def test_start_refuses_a_missing_binary(backend):
    with pytest.raises(FileNotFoundError, match="llama-server not found"):
        backend.start()
