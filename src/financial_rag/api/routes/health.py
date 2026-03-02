"""GET /health — service liveness and model status."""

from __future__ import annotations

import time

from fastapi import APIRouter
from pydantic import BaseModel

from financial_rag.llm import engine

router = APIRouter()

_start_time = time.time()


class HealthResponse(BaseModel):
    status: str
    model: str
    model_loaded: bool
    uptime_s: float


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model=engine.get_model_name(),
        model_loaded=engine.is_loaded(),
        uptime_s=round(time.time() - _start_time, 2),
    )
