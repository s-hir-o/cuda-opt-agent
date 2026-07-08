"""Tests for optimization run API endpoints."""

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


@pytest.fixture
def kernel_id(client) -> int:
    resp = client.post("/api/v1/kernels", json={
        "name": "matmul.cu", "source_code": "// kernel code",
    })
    return resp.json()["id"]


class TestRunEndpoints:

    def test_list_runs_empty(self, client):
        response = client.get("/api/v1/runs")
        assert response.status_code == 200
        assert response.json()["total"] == 0

    def test_start_run(self, client, kernel_id):
        response = client.post("/api/v1/runs", json={
            "kernel_id": kernel_id,
            "max_iterations": 5,
        })
        assert response.status_code == 201
        data = response.json()
        assert data["status"] == "pending"
        assert data["kernel_name"] == "matmul.cu"

    def test_start_run_invalid_kernel(self, client):
        response = client.post("/api/v1/runs", json={
            "kernel_id": 999,
        })
        assert response.status_code == 404

    def test_get_run_not_found(self, client):
        response = client.get("/api/v1/runs/nonexistent")
        assert response.status_code == 404

    def test_start_run_custom_arch(self, client, kernel_id):
        response = client.post("/api/v1/runs", json={
            "kernel_id": kernel_id,
            "arch": "sm_90",
        })
        assert response.status_code == 201
        assert response.json()["target_arch"] == "sm_90"
