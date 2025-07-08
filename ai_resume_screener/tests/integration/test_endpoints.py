"""
Integration Tests for All Endpoints

This module contains comprehensive integration tests for all API endpoints
including error cases and edge scenarios.
"""

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient

from app.main import app


class TestEndpointsIntegration:
    """Integration tests for all endpoints."""
    
    @pytest.mark.asyncio
    async def test_root_endpoint(self, async_client: AsyncClient):
        """Test root endpoint."""
        response = await async_client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "endpoints" in data
    
    @pytest.mark.asyncio
    async def test_invalid_file_upload(self, async_client: AsyncClient):
        """Test upload with invalid file type."""
        files = {"file": ("test.txt", b"Plain text content", "text/plain")}
        
        response = await async_client.post("/api/upload-resume", files=files)
        
        assert response.status_code == 400
        data = response.json()
        assert data["error"] is True
    
    @pytest.mark.asyncio
    async def test_empty_scoring_request(self, async_client: AsyncClient):
        """Test scoring with empty data."""
        scoring_data = {
            "resume_text": "",
            "job_description": ""
        }
        
        response = await async_client.post("/api/score-resume", json=scoring_data)
        
        assert response.status_code == 400
        data = response.json()
        assert data["error"] is True
