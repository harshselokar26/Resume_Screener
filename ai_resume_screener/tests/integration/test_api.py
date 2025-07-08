"""
Integration Tests for API Endpoints

This module contains integration tests for the main API endpoints
testing the complete request-response cycle.
"""

import pytest
import json
import io
from fastapi.testclient import TestClient
from httpx import AsyncClient

from app.main import app
from tests import SAMPLE_RESUME_TEXT, SAMPLE_JOB_DESCRIPTION


class TestAPIIntegration:
    """Integration tests for API endpoints."""
    
    @pytest.mark.asyncio
    async def test_upload_resume_endpoint(self, async_client: AsyncClient):
        """Test resume upload endpoint."""
        # Create a mock PDF file
        pdf_content = b"%PDF-1.4\nMock PDF content"
        files = {"file": ("test_resume.pdf", io.BytesIO(pdf_content), "application/pdf")}
        
        response = await async_client.post("/api/upload-resume", files=files)
        
        assert response.status_code == 201
        data = response.json()
        assert "file_id" in data
        assert "extracted_text" in data
        assert data["status"] == "success"
    
    @pytest.mark.asyncio
    async def test_score_resume_endpoint(self, async_client: AsyncClient):
        """Test resume scoring endpoint."""
        scoring_data = {
            "resume_text": SAMPLE_RESUME_TEXT,
            "job_description": SAMPLE_JOB_DESCRIPTION,
            "include_detailed_analysis": True,
            "include_recommendations": True
        }
        
        response = await async_client.post(
            "/api/score-resume",
            json=scoring_data
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "similarity_score" in data
        assert "matching_skills" in data
        assert "missing_skills" in data
        assert data["status"] == "success"
    
    @pytest.mark.asyncio
    async def test_extract_skills_endpoint(self, async_client: AsyncClient):
        """Test skills extraction endpoint."""
        form_data = {
            "resume_text": SAMPLE_RESUME_TEXT,
            "job_description": SAMPLE_JOB_DESCRIPTION
        }
        
        response = await async_client.post(
            "/api/extract-skills",
            data=form_data
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "technical_skills" in data
        assert "soft_skills" in data
        assert "certifications" in data
        assert data["status"] == "success"
    
    @pytest.mark.asyncio
    async def test_health_endpoint(self, async_client: AsyncClient):
        """Test health check endpoint."""
        response = await async_client.get("/api/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "services" in data
        assert "version" in data
