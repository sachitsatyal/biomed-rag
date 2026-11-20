"""The web API. Run locally:  uv run uvicorn biomed_rag.serving.app:app --port 8000"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from biomed_rag.serving.service import AskRequest, AskResponse, answer_question, load_components

logger = logging.getLogger("biomed_rag.api")


def create_app(load=load_components) -> FastAPI:
    """Build the app. `load` builds the pipeline; tests pass a fake one."""

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

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model": app.state.components.model}

    @app.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest) -> AskResponse:
        try:
            return answer_question(app.state.components, request)
        except Exception as error:
            logger.exception("answering failed")
            raise HTTPException(
                status_code=503, detail=f"answering failed: {type(error).__name__}"
            ) from error

    return app


app = create_app()
