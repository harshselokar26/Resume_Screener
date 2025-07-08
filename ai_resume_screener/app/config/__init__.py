"""
Configuration Package for AI Resume Screener

This package contains all configuration-related modules including
application settings, logging configuration, and environment management.
"""

from app.config.settings import settings
from app.config.logging import setup_logging, get_logger

__all__ = [
    "settings",
    "setup_logging", 
    "get_logger"
]
