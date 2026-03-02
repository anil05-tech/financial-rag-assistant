"""HuggingFace pipeline singleton for text generation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from transformers import pipeline

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a helpful financial assistant. "
    "Answer questions clearly and concisely."
)


@dataclass
class GenerateResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float


# ---------------------------------------------------------------------------
# Module-level singleton state
# ---------------------------------------------------------------------------

_pipeline: Any = None
_model_name: str = ""
_tokenizer: Any = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_engine(model_name: str) -> None:
    """Load the HuggingFace pipeline once at application startup."""
    global _pipeline, _model_name, _tokenizer

    _model_name = model_name
    _pipeline = pipeline(
        "text-generation",
        model=model_name,
    )
    _tokenizer = _pipeline.tokenizer


def get_model_name() -> str:
    return _model_name


def is_loaded() -> bool:
    return _pipeline is not None


def generate(prompt: str, max_new_tokens: int = 256) -> GenerateResult:
    """Run inference and return structured result with token counts and latency."""
    if _pipeline is None or _tokenizer is None:
        raise RuntimeError("Engine not loaded. Call load_engine() first.")

    # Build chat-style prompt using TinyLlama template
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    formatted = _tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    prompt_tokens = len(_tokenizer.encode(formatted))

    t0 = time.perf_counter()
    outputs = _pipeline(
        formatted,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        pad_token_id=_tokenizer.eos_token_id,
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    full_text: str = outputs[0]["generated_text"]
    # Strip the prompt prefix so we only return the new tokens
    response_text = full_text[len(formatted):].strip()

    completion_tokens = len(_tokenizer.encode(response_text))

    return GenerateResult(
        text=response_text,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_ms=round(latency_ms, 2),
    )
