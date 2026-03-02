"""Tests for POST /generate."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from financial_rag.api.app import create_app
from financial_rag.llm.engine import GenerateResult

FAKE_RESULT = GenerateResult(
    text="A P/E ratio measures price relative to earnings.",
    prompt_tokens=20,
    completion_tokens=10,
    latency_ms=123.4,
)


def _patched_client() -> TestClient:
    app = create_app()
    return TestClient(app, raise_server_exceptions=True)


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    # Redirect log writes to a temp dir
    monkeypatch.chdir(tmp_path)
    app = create_app()
    with (
        patch("financial_rag.api.app.engine.load_engine"),
        patch("financial_rag.api.routes.generate.engine.is_loaded", return_value=True),
        patch("financial_rag.api.routes.generate.engine.get_model_name", return_value="test-model"),
        patch("financial_rag.api.routes.generate.engine.generate", return_value=FAKE_RESULT),
    ):
        with TestClient(app) as c:
            yield c


def test_generate_200(client: TestClient) -> None:
    resp = client.post("/generate", json={"prompt": "What is a P/E ratio?"})
    assert resp.status_code == 200


def test_generate_schema(client: TestClient) -> None:
    resp = client.post("/generate", json={"prompt": "What is a P/E ratio?"})
    data = resp.json()
    assert data["response"] == FAKE_RESULT.text
    assert data["prompt_tokens"] == FAKE_RESULT.prompt_tokens
    assert data["completion_tokens"] == FAKE_RESULT.completion_tokens
    assert data["latency_ms"] == FAKE_RESULT.latency_ms


def test_generate_logs_jsonl(client: TestClient, tmp_path: Path) -> None:
    client.post("/generate", json={"prompt": "What is a bond?"})
    log_file = tmp_path / "logs" / "requests.jsonl"
    assert log_file.exists(), "JSONL log file was not created"
    line = json.loads(log_file.read_text(encoding="utf-8").strip())
    assert line["prompt"] == "What is a bond?"
    assert "ts" in line
    assert "latency_ms" in line


def test_generate_503_when_not_loaded() -> None:
    app = create_app()
    with (
        patch("financial_rag.api.app.engine.load_engine"),
        patch("financial_rag.api.routes.generate.engine.is_loaded", return_value=False),
    ):
        with TestClient(app) as client:
            resp = client.post("/generate", json={"prompt": "test"})
    assert resp.status_code == 503


def test_generate_422_empty_prompt(client: TestClient) -> None:
    resp = client.post("/generate", json={"prompt": ""})
    assert resp.status_code == 422
