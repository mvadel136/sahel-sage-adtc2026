"""HTTP surface of the offline console.

Ports app/server.py onto the new modules. Two structural changes:

* The app is built by `create_app(build_context)` instead of importing a
  module-level singleton, so a test can inject a fake backend and a tmp
  library without a subprocess or a port.
* `/api/ask` now streams a fourth event, ``contract``, carrying the parsed
  answer sections. The UI renders those sections rather than a text blob, and
  the ``status`` it carries is the *resolved* one (see service.resolve_status),
  so the caution the pipeline decided on is the caution the reader sees.

Event order on /api/ask is part of the contract with the UI:
``citations`` → ``token``* → ``contract`` → ``done`` (or ``error``).
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable, Iterator
from contextlib import asynccontextmanager, suppress
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from sahel_sage.app.context import AppContext, internet_reachable
from sahel_sage.app.service import AnswerResult, answer_stream, retrieve

UI_FILE = Path(__file__).resolve().parent / "ui" / "index.html"

#: How often strict mode re-verifies that the machine is still air-gapped.
OFFLINE_PROBE_INTERVAL_S = 60.0


class Ask(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    lang: str = "en"
    #: Passages to retrieve. Default 8 since 2026-08-13 — at 4 the coverage
    #: score was length-biased and refused long questions structurally.
    k: int = Field(default=8, ge=0, le=12)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def contract_payload(result: AnswerResult) -> dict:
    """The parsed answer as the UI consumes it.

    ``raw_text`` travels alongside the sections so the UI can fall back to it
    when ``structured`` is false — never show a farmer an empty template.
    """
    c = result.parse.contract
    return {
        "status": str(result.status),
        "structured": result.structured,
        "repaired": result.repaired,
        "likely_issue": c.likely_issue,
        "actions": c.actions,
        "timing": c.timing,
        "caution": c.caution,
        "sources": c.sources,
        "invalid_citations": result.parse.invalid_citations,
        "missing": result.parse.missing,
        "confidence": result.pack.confidence,
        "sufficient": result.pack.sufficient,
        "raw_text": result.raw_text,
    }


async def _offline_watchdog(ctx: AppContext) -> None:
    """Re-probe the network every minute while strict mode is on.

    A start-up check alone would let the badge keep claiming "offline" hours
    after someone plugged in a cable. Only runs under --strict-offline, so a
    normal (or test) run opens no sockets at all.
    """
    while True:
        ctx.internet_seen = await asyncio.to_thread(internet_reachable)
        await asyncio.sleep(OFFLINE_PROBE_INTERVAL_S)


def create_app(build_context: Callable[[], AppContext]) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        ctx = build_context()
        app.state.ctx = ctx
        watchdog = asyncio.create_task(_offline_watchdog(ctx)) if ctx.strict_offline else None
        try:
            yield
        finally:
            if watchdog is not None:
                watchdog.cancel()
                with suppress(asyncio.CancelledError):
                    await watchdog
            ctx.close()

    app = FastAPI(title="Sahel Sage", lifespan=lifespan)

    def context(request: Request) -> AppContext:
        ctx: AppContext | None = getattr(request.app.state, "ctx", None)
        if ctx is None:
            raise HTTPException(503, "model backend not ready")
        return ctx

    @app.get("/")
    def index() -> FileResponse:
        # no-store: the console is one self-contained file that changes with
        # the code. A browser that caches it shows yesterday's UI against
        # today's API — which is exactly how a round of fixes "didn't work"
        # for the first person who tested them.
        return FileResponse(UI_FILE, headers={"Cache-Control": "no-store"})

    @app.get("/api/status")
    def status(request: Request) -> dict:
        ctx = context(request)
        model = Path(ctx.model_path)
        return {
            "model": model.name,
            # A missing GGUF must not 500 the status route: the UI's job is
            # then to say the model is missing, which it cannot do if the
            # status call itself failed.
            "model_size_mb": round(model.stat().st_size / 1e6) if model.exists() else 0,
            "threads": ctx.threads,
            "library": ctx.retriever.stats(),
            "offline_enforced": ctx.offline_enforced,
            "strict_offline": ctx.strict_offline,
            "internet_reachable": ctx.internet_seen,
        }

    @app.get("/api/library")
    def library(request: Request) -> dict:
        return {"documents": context(request).retriever.list_documents()}

    @app.post("/api/search")
    def search(req: Ask, request: Request) -> dict:
        pack = retrieve(req.question, req.lang, req.k or 4, context(request))
        return {
            "citations": [c.to_dict() for c in pack.items],
            "confidence": pack.confidence,
            "sufficient": pack.sufficient,
        }

    @app.post("/api/ask")
    def ask(req: Ask, request: Request) -> StreamingResponse:
        ctx = context(request)

        def events() -> Iterator[str]:
            t0, n = time.time(), 0
            try:
                for kind, payload in answer_stream(req.question, req.lang, req.k, ctx):
                    if kind == "pack":
                        yield _sse("citations", {"citations": [c.to_dict() for c in payload.items]})
                    elif kind == "token":
                        n += 1
                        yield _sse("token", {"t": payload})
                    else:
                        yield _sse("contract", contract_payload(payload))
            except Exception as exc:  # a dead backend must not hang the browser
                yield _sse("error", {"message": str(exc)})
                return
            dt = max(time.time() - t0, 1e-6)
            yield _sse("done", {"tokens": n, "seconds": round(dt, 2), "tps": round(n / dt, 2)})

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app
