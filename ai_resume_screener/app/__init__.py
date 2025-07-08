"""
AI Resume Screener Application Package

This package contains the core FastAPI application for resume screening
and job description matching using NLP techniques.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"
__description__ = "AI-powered resume screening and job matching system"

# Package-level imports for easy access
from app.main import app
from app.config.settings import settings

__all__ = [
    "app",
    "settings",
    "__version__",
    "__author__",
    "__email__",
    "__description__",
]
