"""
API Package for AI Resume Screener

This package contains all API-related modules including routes,
dependencies, middleware, and request/response handling.
"""

from app.api.routes import router
from app.api.dependencies import get_current_user, validate_file_upload
from app.api.middleware import LoggingMiddleware, ErrorHandlingMiddleware, ProcessTimeMiddleware

__all__ = [
    "router",
    "get_current_user",
    "validate_file_upload",
    "LoggingMiddleware",
    "ErrorHandlingMiddleware", 
    "ProcessTimeMiddleware"
]
