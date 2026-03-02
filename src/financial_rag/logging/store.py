"""Async JSONL request logger.

Each call to log_request() appends one JSON line to logs/requests.jsonl.
The logs/ directory is created automatically on first write.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import aiofiles

LOG_PATH = Path("logs") / "requests.jsonl"


@dataclass
class LogRecord:
    model: str
    prompt: str
    response: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int


async def log_request(record: LogRecord) -> None:
    """Append *record* as a single JSON line to LOG_PATH."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "ts": datetime.now(UTC).isoformat(),
        **asdict(record),
    }

    async with aiofiles.open(LOG_PATH, mode="a", encoding="utf-8") as fh:
        await fh.write(json.dumps(entry) + "\n")
