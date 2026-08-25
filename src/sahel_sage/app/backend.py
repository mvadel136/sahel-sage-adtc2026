"""Adapter over the local llama.cpp server.

The app never links against llama.cpp and never imports a Python binding: it
talks HTTP to a ``llama-server`` process it owns. That keeps the boundary
honest, the exact same GGUF the audit measures is the one the app answers
from, and the model can be swapped by changing one path.

Two deliberate departures from the legacy ``app/llm.py``:

* **Raw completions, not chat completions.** The shipped model is Base-trained
  on the raw contract rendering from ``core.prompts.render_raw``. Routing the
  app through ``/v1/chat/completions`` would let llama.cpp's chat template
  wrap the prompt in markers the model never saw in training, a train/serve
  mismatch, which is the most expensive class of scoring defect we can ship.
  So both entry points post to ``/v1/completions`` with a rendered string.
* **The binary is resolved by default_server_binary()** (``lab.audit_server``),
  the audit-parity build, instead of an environment variable read at import.

Everything here is offline by construction: the only host contacted is
127.0.0.1.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Protocol

from sahel_sage.core.config import load_settings

#: Stop strings for raw completions. The model is trained on a single
#: "SAHEL SAGE:" turn, so anything that looks like the start of a new prompt
#: block means it has begun hallucinating the next example.
DEFAULT_STOP: tuple[str, ...] = (
    "\nFARMER'S QUESTION:",
    "\nEXTRACTS FROM THE OFFLINE LIBRARY:",
    "\nSAHEL SAGE:",
)


class ChatBackend(Protocol):
    """Port: anything that can continue a rendered prompt.

    Both methods take the *already rendered* raw prompt (see
    ``core.prompts.render_raw``) rather than a message list, because the raw
    rendering is the contract the model was trained on.
    """

    def complete(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.3,
        stop: Sequence[str] | None = None,
    ) -> str: ...

    def stream(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.3,
        stop: Sequence[str] | None = None,
    ) -> Iterator[str]: ...


def default_server_binary() -> Path:
    """The llama-server to run the console on.

    In order: the audit-parity build from configs/settings.toml (so a demo on
    the development machine runs the same binary the competition measures),
    then $SAHEL_LLAMA_SERVER, then whatever `llama-server` is on PATH. The
    fallbacks exist because the settings path is this machine's measurement
    lab: a judge cloning the public repository does not have it, and "file not
    found" from a console the submission calls load-bearing is the wrong first
    impression.
    """
    # An explicit override outranks the config: SAHEL_LLAMA_SERVER exists so
    # the console can run on a NATIVE build. The audit build keeps all SIMD
    # off to mirror the competition sandbox: right for scored numbers, but a
    # 10-20x reading-speed handicap no real deployment would accept.
    env = os.environ.get("SAHEL_LLAMA_SERVER")
    if env and Path(env).expanduser().exists():
        return Path(env).expanduser()
    configured = load_settings().lab.path("audit_server")
    if configured.exists():
        return configured
    found = shutil.which("llama-server")
    if found:
        return Path(found)
    raise FileNotFoundError(
        "No llama-server found. Install llama.cpp (https://github.com/ggml-org/llama.cpp"
        "/releases), or point SAHEL_LLAMA_SERVER at the binary."
    )


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@dataclass
class LlamaServerConfig:
    model: Path
    binary: Path = field(default_factory=default_server_binary)
    n_ctx: int = 4096
    threads: int = max(1, (os.cpu_count() or 4) // 2)
    port: int = 0  # 0 = pick a free one
    extra_args: tuple[str, ...] = ()
    startup_timeout_s: float = 120.0


class LlamaServerBackend:
    """Owns a ``llama-server`` subprocess and speaks its completions API."""

    def __init__(self, cfg: LlamaServerConfig):
        self.cfg = cfg
        self.port = cfg.port or _free_port()
        self.proc: subprocess.Popen | None = None
        self._log: IO[bytes] | None = None

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        if not self.cfg.binary.exists():
            raise FileNotFoundError(f"llama-server not found: {self.cfg.binary}")
        if not self.cfg.model.exists():
            raise FileNotFoundError(
                f"model not found: {self.cfg.model}\nRun ./download_model.sh first."
            )
        # Opened here, not in __init__, so constructing a config never leaves a
        # stray log file behind on a machine that only inspects the app.
        self._log = (self.cfg.model.parent / "llama-server.log").open("ab")
        cmd = [
            str(self.cfg.binary),
            "-m", str(self.cfg.model),
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "-c", str(self.cfg.n_ctx),
            "-t", str(self.cfg.threads),
            "-ngl", "0",
            "--no-webui",
            *self.cfg.extra_args,
        ]
        # start_new_session: the server gets its own process group so stop()
        # can kill the whole group, not just the parent.
        self.proc = subprocess.Popen(
            cmd, stdout=self._log, stderr=self._log, start_new_session=True
        )
        self._wait_ready()

    def _wait_ready(self) -> None:
        deadline = time.time() + self.cfg.startup_timeout_s
        while time.time() < deadline:
            if self.proc and self.proc.poll() is not None:
                raise RuntimeError(
                    f"llama-server exited with code {self.proc.returncode}; see llama-server.log"
                )
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{self.port}/health", timeout=2
                ) as r:
                    if r.status == 200:
                        return
            except (urllib.error.URLError, OSError):
                time.sleep(0.4)
        raise TimeoutError("llama-server did not become ready in time")

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
        if self._log is not None:
            self._log.close()
            self._log = None

    def __enter__(self) -> LlamaServerBackend:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # -- inference ---------------------------------------------------------
    def _payload(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        stop: Sequence[str] | None,
        stream: bool,
    ) -> dict:
        return {
            "prompt": prompt,
            "stream": stream,
            "temperature": temperature,
            "top_p": 0.9,
            "max_tokens": max_tokens,
            "stop": list(DEFAULT_STOP if stop is None else stop),
            # The evidence block dominates the prompt and barely changes
            # between the first attempt and its repair; reusing the KV cache is
            # the difference between a usable and an unusable repair latency.
            "cache_prompt": True,
        }

    def _post(self, payload: dict, stream: bool):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/v1/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        return urllib.request.urlopen(req, timeout=600 if stream else 300)

    def complete(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.3,
        stop: Sequence[str] | None = None,
    ) -> str:
        payload = self._payload(prompt, max_tokens, temperature, stop, stream=False)
        with self._post(payload, stream=False) as resp:
            body = json.loads(resp.read())
        return body["choices"][0]["text"]

    def stream(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.3,
        stop: Sequence[str] | None = None,
    ) -> Iterator[str]:
        payload = self._payload(prompt, max_tokens, temperature, stop, stream=True)
        with self._post(payload, stream=True) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    return
                try:
                    piece = json.loads(data)["choices"][0].get("text", "")
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
                if piece:
                    yield piece
