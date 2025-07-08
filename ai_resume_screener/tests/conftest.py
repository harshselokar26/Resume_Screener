"""
Pytest Configuration and Shared Fixtures

This module contains pytest configuration and shared fixtures
used across all test modules.
"""

import os
import tempfile
import shutil
from pathlib import Path
import pytest
import asyncio
from typing import Generator, Dict, Any
from unittest.mock import Mock, AsyncMock

# FastAPI testing
from fastapi.testclient import TestClient
from httpx import AsyncClient

# Application imports
from app.main import app
from app.config.settings import settings
from app.services.pdf_parser import PDFParser
from app.services.nlp_processor import NLPProcessor
from app.services.scorer import ResumeScorer
from app.services.file_handler import FileHandler
from app.models.database import Base, create_tables
from tests import (
    TEST_DATABASE_URL, 
    TEST_UPLOAD_DIR, 
    SAMPLE_RESUME_TEXT, 
    SAMPLE_JOB_DESCRIPTION
)


# ===== PYTEST CONFIGURATION =====

def pytest_configure(config):
    """Configure pytest settings."""
    # Set test environment
    os.environ["ENVIRONMENT"] = "testing"
    os.environ["DATABASE_URL"] = TEST_DATABASE_URL
    os.environ["UPLOAD_DIR"] = TEST_UPLOAD_DIR


def pytest_unconfigure(config):
    """Cleanup after all tests."""
    # Clean up test files
    if os.path.exists(TEST_UPLOAD_DIR):
        shutil.rmtree(TEST_UPLOAD_DIR)


# ===== EVENT LOOP FIXTURE =====

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ===== APPLICATION FIXTURES =====

@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """
    Create a test client for the FastAPI application.
    
    Yields:
        TestClient: FastAPI test client
    """
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
async def async_client() -> Generator[AsyncClient, None, None]:
    """
    Create an async test client for the FastAPI application.
    
    Yields:
        AsyncClient: Async HTTP client
    """
    async with AsyncClient(app=app, base_url="http://test") as async_test_client:
        yield async_test_client


# ===== TEMPORARY DIRECTORY FIXTURES =====

@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    """
    Create a temporary directory for tests.
    
    Yields:
        str: Path to temporary directory
    """
    temp_directory = tempfile.mkdtemp()
    yield temp_directory
    shutil.rmtree(temp_directory)


@pytest.fixture
def upload_dir(temp_dir: str) -> str:
    """
    Create upload directory for file tests.
    
    Args:
        temp_dir: Temporary directory fixture
        
    Returns:
        str: Path to upload directory
    """
    upload_path = os.path.join(temp_dir, "uploads")
    os.makedirs(upload_path, exist_ok=True)
    return upload_path


# ===== SERVICE FIXTURES =====

@pytest.fixture
def pdf_parser() -> PDFParser:
    """
    Create PDF parser instance for testing.
    
    Returns:
        PDFParser: PDF parser service
    """
    return PDFParser()


@pytest.fixture
def nlp_processor() -> NLPProcessor:
    """
    Create NLP processor instance for testing.
    
    Returns:
        NLPProcessor: NLP processor service
    """
    return NLPProcessor()


@pytest.fixture
def resume_scorer() -> ResumeScorer:
    """
    Create resume scorer instance for testing.
    
    Returns:
        ResumeScorer: Resume scorer service
    """
    return ResumeScorer()


@pytest.fixture
def file_handler(upload_dir: str) -> FileHandler:
    """
    Create file handler instance for testing.
    
    Args:
        upload_dir: Upload directory fixture
        
    Returns:
        FileHandler: File handler service
    """
    # Mock settings for test
    original_upload_dir = settings.UPLOAD_DIR
    settings.UPLOAD_DIR = upload_dir
    
    handler = FileHandler()
    
    yield handler
    
    # Restore original settings
    settings.UPLOAD_DIR = original_upload_dir


# ===== MOCK FIXTURES =====

@pytest.fixture
def mock_nlp_processor():
    """
    Create mock NLP processor for testing.
    
    Returns:
        Mock: Mocked NLP processor
    """
    mock = Mock(spec=NLPProcessor)
    
    # Mock async methods
    mock.extract_skills = AsyncMock(return_value={
        "technical_skills": ["Python", "JavaScript", "React"],
        "soft_skills": ["Leadership", "Communication"],
        "certifications": []
    })
    
    mock.extract_experience = AsyncMock(return_value={
        "total_years": 5,
        "details": [{"text": "5 years experience", "years": 5}]
    })
    
    mock.extract_education = AsyncMock(return_value=[
        {"text": "Bachelor's in Computer Science", "type": "education"}
    ])
    
    mock.extract_keywords = AsyncMock(return_value=[
        "software", "engineer", "python", "javascript", "react"
    ])
    
    mock.health_check = AsyncMock(return_value=True)
    
    return mock


@pytest.fixture
def mock_pdf_parser():
    """
    Create mock PDF parser for testing.
    
    Returns:
        Mock: Mocked PDF parser
    """
    mock = Mock(spec=PDFParser)
    
    mock.extract_text_from_pdf = AsyncMock(return_value=SAMPLE_RESUME_TEXT)
    mock.extract_text_from_doc = AsyncMock(return_value=SAMPLE_RESUME_TEXT)
    mock.extract_metadata = AsyncMock(return_value={
        "file_size": 1024,
        "page_count": 1,
        "author": "John Doe"
    })
    mock.validate_file = AsyncMock(return_value=True)
    mock.health_check = AsyncMock(return_value=True)
    
    return mock


@pytest.fixture
def mock_resume_scorer():
    """
    Create mock resume scorer for testing.
    
    Returns:
        Mock: Mocked resume scorer
    """
    mock = Mock(spec=ResumeScorer)
    
    mock.calculate_similarity = AsyncMock(return_value=0.85)
    mock.find_matching_skills = AsyncMock(return_value=[
        "Python", "JavaScript", "React"
    ])
    mock.find_missing_skills = AsyncMock(return_value=[
        "Docker", "Kubernetes"
    ])
    mock.get_detailed_analysis = AsyncMock(return_value={
        "overall_similarity": 0.85,
        "skills_similarity": 0.75,
        "strengths": ["Strong technical skills"],
        "weaknesses": ["Limited cloud experience"]
    })
    mock.get_recommendation = AsyncMock(return_value={
        "decision": "Recommended",
        "confidence": "High",
        "reasons": ["Good skill match"],
        "suggestions": ["Gain cloud experience"]
    })
    mock.health_check = AsyncMock(return_value=True)
    
    return mock


# ===== DATA FIXTURES =====

@pytest.fixture
def sample_resume_data() -> Dict[str, Any]:
    """
    Sample resume data for testing.
    
    Returns:
        Dict: Sample resume data
    """
    return {
        "original_filename": "john_doe_resume.pdf",
        "file_size": 1024,
        "extracted_text": SAMPLE_RESUME_TEXT,
        "text_length": len(SAMPLE_RESUME_TEXT),
        "basic_info": {
            "emails": ["john.doe@example.com"],
            "phones": ["(555) 123-4567"],
            "names": ["John Doe"]
        }
    }


@pytest.fixture
def sample_job_description_data() -> Dict[str, Any]:
    """
    Sample job description data for testing.
    
    Returns:
        Dict: Sample job description data
    """
    return {
        "job_title": "Senior Software Engineer",
        "company_name": "TechCorp",
        "job_description": SAMPLE_JOB_DESCRIPTION,
        "job_level": "senior"
    }


@pytest.fixture
def sample_scoring_request() -> Dict[str, Any]:
    """
    Sample scoring request data for testing.
    
    Returns:
        Dict: Sample scoring request
    """
    return {
        "resume_text": SAMPLE_RESUME_TEXT,
        "job_description": SAMPLE_JOB_DESCRIPTION,
        "include_detailed_analysis": True,
        "include_recommendations": True
    }


@pytest.fixture
def sample_skills_data() -> Dict[str, Any]:
    """
    Sample skills data for testing.
    
    Returns:
        Dict: Sample skills data
    """
    return {
        "technical_skills": [
            "Python", "JavaScript", "React", "Node.js", 
            "PostgreSQL", "MongoDB", "AWS", "Docker"
        ],
        "soft_skills": [
            "Leadership", "Communication", "Problem Solving", "Teamwork"
        ],
        "certifications": [
            "AWS Certified Solutions Architect"
        ]
    }


# ===== FILE FIXTURES =====

@pytest.fixture
def sample_pdf_file(temp_dir: str) -> str:
    """
    Create a sample PDF file for testing.
    
    Args:
        temp_dir: Temporary directory fixture
        
    Returns:
        str: Path to sample PDF file
    """
    pdf_content = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n"
    pdf_path = os.path.join(temp_dir, "sample_resume.pdf")
    
    with open(pdf_path, "wb") as f:
        f.write(pdf_content)
    
    return pdf_path


@pytest.fixture
def sample_text_file(temp_dir: str) -> str:
    """
    Create a sample text file for testing.
    
    Args:
        temp_dir: Temporary directory fixture
        
    Returns:
        str: Path to sample text file
    """
    text_path = os.path.join(temp_dir, "sample_jd.txt")
    
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(SAMPLE_JOB_DESCRIPTION)
    
    return text_path


@pytest.fixture
def invalid_file(temp_dir: str) -> str:
    """
    Create an invalid file for testing error cases.
    
    Args:
        temp_dir: Temporary directory fixture
        
    Returns:
        str: Path to invalid file
    """
    invalid_path = os.path.join(temp_dir, "invalid_file.exe")
    
    with open(invalid_path, "wb") as f:
        f.write(b"This is not a valid resume file")
    
    return invalid_path


# ===== DATABASE FIXTURES =====

@pytest.fixture
def test_db():
    """
    Create test database for integration tests.
    
    Yields:
        Database engine
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    
    # Create test database
    engine = create_engine(TEST_DATABASE_URL, echo=False)
    create_tables(engine)
    
    # Create session
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    yield TestingSessionLocal
    
    # Cleanup
    Base.metadata.drop_all(bind=engine)


# ===== AUTHENTICATION FIXTURES =====

@pytest.fixture
def auth_headers() -> Dict[str, str]:
    """
    Create authentication headers for testing.
    
    Returns:
        Dict: Authentication headers
    """
    return {
        "Authorization": "Bearer test_token_123",
        "Content-Type": "application/json"
    }


@pytest.fixture
def api_key_headers() -> Dict[str, str]:
    """
    Create API key headers for testing.
    
    Returns:
        Dict: API key headers
    """
    return {
        "X-API-Key": "rsc_test_api_key_123",
        "Content-Type": "application/json"
    }


# ===== PERFORMANCE FIXTURES =====

@pytest.fixture
def performance_timer():
    """
    Timer fixture for performance testing.
    
    Yields:
        Function to measure execution time
    """
    import time
    
    def timer():
        start_time = time.time()
        return lambda: time.time() - start_time
    
    yield timer


# ===== CLEANUP FIXTURES =====

@pytest.fixture(autouse=True)
def cleanup_test_files():
    """
    Automatically cleanup test files after each test.
    """
    yield
    
    # Cleanup test upload directory
    if os.path.exists(TEST_UPLOAD_DIR):
        for file in os.listdir(TEST_UPLOAD_DIR):
            file_path = os.path.join(TEST_UPLOAD_DIR, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception:
                pass


# ===== PARAMETRIZED FIXTURES =====

@pytest.fixture(params=[
    "sample_resume_1.pdf",
    "sample_resume_2.docx", 
    "sample_resume_3.txt"
])
def resume_filename(request):
    """
    Parametrized fixture for different resume file types.
    
    Returns:
        str: Resume filename
    """
    return request.param


@pytest.fixture(params=[0.1, 0.5, 0.8, 0.95])
def similarity_score(request):
    """
    Parametrized fixture for different similarity scores.
    
    Returns:
        float: Similarity score
    """
    return request.param


# ===== UTILITY FIXTURES =====

@pytest.fixture
def assert_timing():
    """
    Fixture for asserting execution time.
    
    Returns:
        Function to assert timing
    """
    def _assert_timing(func, max_time: float):
        import time
        start = time.time()
        result = func()
        elapsed = time.time() - start
        assert elapsed < max_time, f"Function took {elapsed:.2f}s, expected < {max_time}s"
        return result
    
    return _assert_timing


@pytest.fixture
def mock_settings():
    """
    Mock settings for testing.
    
    Returns:
        Mock settings object
    """
    mock = Mock()
    mock.UPLOAD_DIR = TEST_UPLOAD_DIR
    mock.MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    mock.ALLOWED_FILE_TYPES = ["pdf", "doc", "docx"]
    mock.SPACY_MODEL = "en_core_web_sm"
    mock.SECRET_KEY = "test_secret_key"
    mock.ENVIRONMENT = "testing"
    
    return mock
