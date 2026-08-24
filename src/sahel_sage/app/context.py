"""Everything a request needs, in one object.

The legacy server kept a module-level ``STATE`` dict, which made the answer
pipeline untestable without booting a subprocess and made the "is the backend
ready?" question a `dict.get` away from a 500. AppContext is built once in the
FastAPI lifespan, attached to ``app.state``, and passed explicitly into the
pipeline — so tests hand it a fake backend and a tmp library instead.
"""

from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from pathlib import Path

from sahel_sage.app.backend import default_server_binary, ChatBackend, LlamaServerBackend, LlamaServerConfig
from sahel_sage.core.config import load_retrieval, load_settings
from sahel_sage.retrieval.store import NullRetriever, Retriever, open_retriever


def internet_reachable(timeout: float = 2.0) -> bool:
    """True if a DNS port on a public resolver accepts a TCP connection.

    Deliberately a raw TCP probe rather than a DNS or HTTP request: it answers
    "could this machine exfiltrate anything?" without itself sending a query.
    """
    for host, port in (("1.1.1.1", 53), ("8.8.8.8", 53)):
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except OSError:
            continue
    return False


@dataclass
class AppContext:
    """Live state of one running console."""

    cfg: LlamaServerConfig | None
    backend: ChatBackend
    retriever: Retriever | NullRetriever
    model_path: Path
    threads: int
    strict_offline: bool
    #: Sufficiency threshold on IDF-weighted coverage, from
    #: configs/retrieval.toml. Deliberately has no default: below it the app
    #: refuses to answer, so a construction site that forgets to supply it must
    #: fail at once rather than fall back to an uncalibrated constant.
    confidence_threshold: float
    #: Passages retrieved, scored and shown. 8 since 2026-08-13; see
    #: configs/retrieval.toml for the measurement behind it.
    k: int = 8
    #: Result of the most recent probe; None when no probe has run (the
    #: monitor only runs under --strict-offline, so "unknown" is the honest
    #: value everywhere else — never report an unverified claim as verified).
    internet_seen: bool | None = None

    @property
    def offline_enforced(self) -> bool:
        """Strict mode requested *and* no probe has found the internet.

        If the network comes back while the console runs, this flips to False
        and the UI badge disappears: the badge is a live claim, not a flag.
        """
        return self.strict_offline and not self.internet_seen

    def close(self) -> None:
        stop = getattr(self.backend, "stop", None)
        if callable(stop):
            stop()


def default_threads() -> int:
    return max(1, (os.cpu_count() or 4) // 2)


def build_context(
    model: Path | None = None,
    db: Path | None = None,
    threads: int | None = None,
    # Eight passages at the raised context budget render to ~3.1k tokens, which
    # leaves no headroom at 4096 for a 512-token answer. The supporting app
    # stack is explicitly exempt from the competition's memory and speed
    # measurement, so the extra KV cache costs us nothing that is scored.
    n_ctx: int = 8192,
    strict_offline: bool = False,
) -> AppContext:
    """Boot the real backend. Called from the FastAPI lifespan, never at import.

    Unset paths fall back to configs/settings.toml so that the console, the
    evaluation harness and the audit all point at the same GGUF and the same
    library database.
    """
    settings = load_settings()
    retrieval = load_retrieval()
    model_path = Path(model) if model else settings.repo.abs_model_path()
    db_path = Path(db) if db else settings.repo.abs_library_db()
    cfg = LlamaServerConfig(
        model=model_path,
        # Resolved, not hardcoded: SAHEL_LLAMA_SERVER > configured audit
        # build > PATH. The hardcoded lab path here was both why the env
        # override silently did nothing and a second judge-machine dead end.
        binary=default_server_binary(),
        n_ctx=n_ctx,
        threads=threads or default_threads(),
    )
    backend = LlamaServerBackend(cfg)
    backend.start()
    return AppContext(
        cfg=cfg,
        backend=backend,
        retriever=open_retriever(db_path),
        model_path=model_path,
        threads=cfg.threads,
        strict_offline=strict_offline,
        confidence_threshold=retrieval.confidence.threshold,
        k=retrieval.retrieval.k,
    )
