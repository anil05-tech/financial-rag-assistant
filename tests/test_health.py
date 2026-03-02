"""Tests for GET /health."""

from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient

from financial_rag.api.app import create_app


def _make_client() -> TestClient:
    """Create a TestClient with model loading suppressed."""
    app = create_app()
    # Suppress lifespan (model load) — not needed for health tests
    with patch("financial_rag.api.app.engine.load_engine"):
        with TestClient(app, raise_server_exceptions=True) as client:
            return client


def test_health_returns_200() -> None:
    with patch("financial_rag.api.app.engine.load_engine"):
        with TestClient(create_app()) as client:
            resp = client.get("/health")
    assert resp.status_code == 200


def test_health_schema() -> None:
    with patch("financial_rag.api.app.engine.load_engine"):
        with TestClient(create_app()) as client:
            data = client.get("/health").json()
    assert data["status"] == "ok"
    assert "model" in data
    assert "model_loaded" in data
    assert "uptime_s" in data
    assert isinstance(data["uptime_s"], float)
