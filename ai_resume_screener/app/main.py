"""
FastAPI Main Application Entry Point

This module initializes the FastAPI application, configures middleware,
includes routers, and sets up the core application structure.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import os
from typing import AsyncGenerator

# Internal imports
from app.config.settings import settings
from app.config.logging import setup_logging
from app.api.routes import router as api_router
from app.api.middleware import (
    LoggingMiddleware,
    ErrorHandlingMiddleware,
    ProcessTimeMiddleware
)
from app.utils.exceptions import ResumeScreenerException

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager for startup and shutdown events.
    
    This function handles:
    - Application startup: Initialize resources, download models
    - Application shutdown: Cleanup resources, close connections
    """
    # Startup
    logger.info("🚀 Starting AI Resume Screener application...")
    
    # Create necessary directories
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Download spaCy models if not present
    try:
        import spacy
        try:
            spacy.load("en_core_web_sm")
            logger.info("✅ spaCy model 'en_core_web_sm' loaded successfully")
        except OSError:
            logger.warning("⚠️ spaCy model 'en_core_web_sm' not found. Please install it.")
            logger.info("Run: python -m spacy download en_core_web_sm")
    except ImportError:
        logger.error("❌ spaCy not installed. Please install it: pip install spacy")
    
    logger.info("✅ Application startup completed")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down AI Resume Screener application...")
    logger.info("✅ Application shutdown completed")


# Initialize FastAPI application
app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.PROJECT_DESCRIPTION,
    version=settings.VERSION,
    docs_url="/docs" if settings.ENVIRONMENT == "development" else None,
    redoc_url="/redoc" if settings.ENVIRONMENT == "development" else None,
    openapi_url="/openapi.json" if settings.ENVIRONMENT == "development" else None,
    lifespan=lifespan
)

# Add Security Middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.ALLOWED_HOSTS
)

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Add Custom Middleware (order matters - last added runs first)
app.add_middleware(ProcessTimeMiddleware)
app.add_middleware(ErrorHandlingMiddleware)
app.add_middleware(LoggingMiddleware)


# Global Exception Handler
@app.exception_handler(ResumeScreenerException)
async def resume_screener_exception_handler(request, exc: ResumeScreenerException):
    """Handle custom application exceptions"""
    logger.error(f"Application error: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "message": exc.detail,
            "error_code": exc.error_code,
            "timestamp": exc.timestamp.isoformat()
        }
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Handle HTTP exceptions"""
    logger.error(f"HTTP error: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "message": exc.detail,
            "status_code": exc.status_code
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """Handle unexpected exceptions"""
    logger.error(f"Unexpected error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "message": "Internal server error occurred",
            "detail": str(exc) if settings.ENVIRONMENT == "development" else None
        }
    )


# Root endpoint
@app.get("/", tags=["Health"])
async def root():
    """
    Root endpoint providing basic application information.
    
    Returns:
        dict: Application status and basic information
    """
    return {
        "message": "🤖 AI Resume Screener API",
        "version": settings.VERSION,
        "status": "active",
        "environment": settings.ENVIRONMENT,
        "docs_url": "/docs" if settings.ENVIRONMENT == "development" else "disabled",
        "endpoints": {
            "health": "/health",
            "upload_resume": "/api/upload-resume",
            "score_resume": "/api/score-resume",
            "extract_skills": "/api/extract-skills"
        }
    }


# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint for monitoring and load balancers.
    
    Returns:
        dict: Application health status and system information
    """
    try:
        # Check spaCy model availability
        import spacy
        spacy_status = "available"
        try:
            nlp = spacy.load("en_core_web_sm")
            spacy_status = "loaded"
        except OSError:
            spacy_status = "model_missing"
    except ImportError:
        spacy_status = "not_installed"
    
    # Check directory permissions
    directories_status = {}
    for directory in ["uploads", "logs", "models"]:
        directories_status[directory] = {
            "exists": os.path.exists(directory),
            "writable": os.access(directory, os.W_OK) if os.path.exists(directory) else False
        }
    
    return {
        "status": "healthy",
        "timestamp": settings.get_current_timestamp(),
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "services": {
            "spacy": spacy_status,
            "directories": directories_status
        },
        "uptime": "Available in production version"
    }


# Include API routes
app.include_router(
    api_router,
    prefix="/api",
    tags=["Resume Screening"]
)


# Additional metadata for OpenAPI documentation
if settings.ENVIRONMENT == "development":
    app.openapi_tags = [
        {
            "name": "Health",
            "description": "Health check and system status endpoints"
        },
        {
            "name": "Resume Screening",
            "description": "Resume upload, parsing, and scoring operations"
        }
    ]


# Development server runner
if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"🚀 Starting development server on {settings.HOST}:{settings.PORT}")
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.ENVIRONMENT == "development",
        log_level=settings.LOG_LEVEL.lower(),
        access_log=True
    )
