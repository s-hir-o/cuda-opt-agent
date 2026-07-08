"""Tests for kernel API endpoints."""

import pytest
from fastapi.testclient import TestClient

from cuda_opt_agent.api.app import create_app
from cuda_opt_agent.core.config import AppSettings
from cuda_opt_agent.memory.database import Base, init_db, dispose_engine


@pytest.fixture
def client(tmp_path):
    """Create a test client with an in-memory database."""
    settings = AppSettings()
    settings.database.url = "sqlite:///:memory:"
    settings.database.echo = False

    init_db(settings.database)
    app = create_app(settings)
    client = TestClient(app)

    yield client

    dispose_engine()


class TestKernelEndpoints:

    def test_submit_kernel(self, client):
        response = client.post("/api/v1/kernels", json={
            "name": "test.cu",
            "source_code": "__global__ void k() {}",
            "language": "cuda",
            "entry_point": "k",
        })
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "test.cu"
        assert data["language"] == "cuda"
        assert data["entry_point"] == "k"

    def test_submit_invalid_language(self, client):
        response = client.post("/api/v1/kernels", json={
            "name": "test.cu",
            "source_code": "// code",
            "language": "fortran",
        })
        assert response.status_code == 400

    def test_list_kernels_empty(self, client):
        response = client.get("/api/v1/kernels")
        assert response.status_code == 200
        data = response.json()
        assert data["kernels"] == []
        assert data["total"] == 0

    def test_submit_and_list(self, client):
        client.post("/api/v1/kernels", json={
            "name": "a.cu", "source_code": "// a",
        })
        client.post("/api/v1/kernels", json={
            "name": "b.cu", "source_code": "// b",
        })

        response = client.get("/api/v1/kernels")
        data = response.json()
        assert data["total"] == 2

    def test_get_kernel_by_id(self, client):
        resp = client.post("/api/v1/kernels", json={
            "name": "test.cu", "source_code": "// code",
        })
        kernel_id = resp.json()["id"]

        response = client.get(f"/api/v1/kernels/{kernel_id}")
        assert response.status_code == 200
        assert response.json()["name"] == "test.cu"

    def test_get_kernel_not_found(self, client):
        response = client.get("/api/v1/kernels/999")
        assert response.status_code == 404

    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
