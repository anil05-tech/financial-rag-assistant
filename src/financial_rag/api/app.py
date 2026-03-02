"""FastAPI application factory."""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from financial_rag.api.routes import generate, health
from financial_rag.llm import engine

DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def _get_model_name() -> str:
    return os.environ.get("MODEL_NAME", DEFAULT_MODEL)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load the LLM once at startup; release on shutdown."""
    model_name = _get_model_name()
    print(f"[startup] Loading model: {model_name}")
    engine.load_engine(model_name)
    print("[startup] Model ready.")
    yield
    print("[shutdown] Bye.")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Financial RAG Assistant — LLM Service",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(health.router)
    app.include_router(generate.router)
    return app


# Module-level instance so uvicorn can reference "financial_rag.api.app:app"
app = create_app()
