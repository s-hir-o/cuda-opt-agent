"""Tests for results API endpoints."""

import pytest
from fastapi.testclient import TestClient

from cuda_opt_agent.api.app import create_app
from cuda_opt_agent.core.config import AppSettings
from cuda_opt_agent.memory.database import init_db, dispose_engine


@pytest.fixture
def client(tmp_path):
    settings = AppSettings()
    settings.database.url = "sqlite:///:memory:"
    init_db(settings.database)
    app = create_app(settings)
    client = TestClient(app)
    yield client
    dispose_engine()


class TestResultsEndpoints:

    def test_metrics_not_found(self, client):
        response = client.get("/api/v1/results/metrics/nonexistent")
        assert response.status_code == 404

    def test_benchmarks_not_found(self, client):
        response = client.get("/api/v1/results/benchmarks/nonexistent")
        assert response.status_code == 404

    def test_history_empty(self, client):
        response = client.get("/api/v1/results/history")
        assert response.status_code == 200
        assert response.json() == []

    def test_compilation_not_implemented(self, client):
        response = client.get("/api/v1/results/compilation/v1")
        assert response.status_code == 501
