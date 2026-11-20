"""The web API. Run locally:  uv run uvicorn biomed_rag.serving.app:app --port 8000"""

from __future__ import annotations

import logging
import math
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse

from biomed_rag.serving.monitoring import Monitor, RequestRecord, question_hash, setup_logging
from biomed_rag.serving.rate_limit import RateLimiter, client_id
from biomed_rag.serving.service import AskRequest, AskResponse, answer_question, load_components

logger = logging.getLogger("biomed_rag.api")
INDEX_HTML = Path(__file__).parent / "static" / "index.html"


def create_app(load=load_components, limiter: RateLimiter | None = None) -> FastAPI:
    """Build the app. `load` builds the pipeline; tests pass fakes and small limits."""
    setup_logging()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.components = load()  # slow work happens once, at startup
        yield

    app = FastAPI(
        title="biomed-rag",
        version="0.1.0",
        description="Cited answers from Alzheimer's disease literature (research prototype, "
        "not for clinical use).",
        lifespan=lifespan,
    )
    app.state.monitor = Monitor()
    app.state.limiter = limiter or RateLimiter.from_env()

    @app.get("/", include_in_schema=False)
    def home() -> FileResponse:
        return FileResponse(INDEX_HTML)  # the web page; it calls /ask from the browser

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model": app.state.components.model}

    @app.get("/stats")
    def stats() -> dict:
        """Rolling usage stats since the last restart (latency, errors, abstentions)."""
        return app.state.monitor.stats()

    @app.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest, http_request: Request) -> AskResponse:
        blocked = app.state.limiter.check(client_id(http_request))
        if blocked:
            app.state.monitor.rate_limited += 1
            retry = math.ceil(blocked.retry_after)
            raise HTTPException(
                status_code=429,
                detail=f"{blocked.reason}, please try again in {retry} s",
                headers={"Retry-After": str(retry)},
            )
        start = time.perf_counter()
        base = {
            "mode": request.mode,
            "question_chars": len(request.question),
            "question_hash": question_hash(request.question),
        }
        try:
            response = answer_question(app.state.components, request)
        except Exception as error:
            logger.exception("answering failed")
            seconds = round(time.perf_counter() - start, 2)
            app.state.monitor.record(
                RequestRecord(**base, seconds=seconds, error=type(error).__name__)
            )
            raise HTTPException(
                status_code=503, detail=f"answering failed: {type(error).__name__}"
            ) from error
        app.state.monitor.record(
            RequestRecord(
                **base,
                seconds=response.seconds,
                tokens=response.tokens,
                abstained=response.abstained,
                used_live=any(e.source == "pubmed" for e in response.evidence),
            )
        )
        return response

    return app


app = create_app()
