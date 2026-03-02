"""POST /generate — run inference and log the request."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from financial_rag.llm import engine
from financial_rag.logging.store import LogRecord, log_request

router = APIRouter()


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4096)
    max_new_tokens: int = Field(default=256, ge=1, le=2048)


class GenerateResponse(BaseModel):
    response: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float


@router.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest) -> GenerateResponse:
    if not engine.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    result = engine.generate(
        prompt=request.prompt,
        max_new_tokens=request.max_new_tokens,
    )

    await log_request(
        LogRecord(
            model=engine.get_model_name(),
            prompt=request.prompt,
            response=result.text,
            latency_ms=result.latency_ms,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
        )
    )

    return GenerateResponse(
        response=result.text,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        latency_ms=result.latency_ms,
    )
