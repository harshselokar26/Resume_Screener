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
from datetime import datetime
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


def get_base_url() -> str:
    """
    Generate base URL for the application.
    
    Returns:
        str: Complete base URL including protocol, host, and port
    """
    protocol = "https" if settings.ENVIRONMENT == "production" else "http"
    
    # Handle different host configurations
    if settings.HOST in ["0.0.0.0", "127.0.0.1"]:
        host = "localhost"
    else:
        host = settings.HOST
    
    # Don't include port for standard ports in production
    if settings.ENVIRONMENT == "production" and settings.PORT in [80, 443]:
        return f"{protocol}://{host}"
    else:
        return f"{protocol}://{host}:{settings.PORT}"


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
    os.makedirs("data", exist_ok=True)
    
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
    
    # Log development server information with proper URLs
    base_url = get_base_url()
    logger.info(f"📚 API Documentation available at:")
    logger.info(f"   • Swagger UI: {base_url}/docs")
    logger.info(f"   • ReDoc: {base_url}/redoc")
    logger.info(f"   • OpenAPI JSON: {base_url}/openapi.json")
    logger.info(f"   • Health Check: {base_url}/health")
    logger.info(f"   • Root Endpoint: {base_url}/")
    logger.info(f"   • API Base: {base_url}/api")
    
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
    lifespan=lifespan,
    contact={
        "name": "AI Resume Screener Support",
        "email": "support@resumescreener.com",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
    servers=[
        {
            "url": get_base_url(),
            "description": f"{settings.ENVIRONMENT.title()} server"
        }
    ]
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
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
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
    logger.error(f"Application error: {exc.message}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "message": exc.message,
            "error_code": exc.error_code,
            "timestamp": exc.timestamp.isoformat(),
            "details": exc.details
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
            "status_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat()
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
            "detail": str(exc) if settings.ENVIRONMENT == "development" else None,
            "timestamp": datetime.utcnow().isoformat()
        }
    )


# Root endpoint with documentation links
@app.get("/", tags=["Health"])
async def root():
    """
    Root endpoint providing basic application information and documentation links.
    
    Returns:
        dict: Application status, basic information, and documentation URLs
    """
    base_url = get_base_url()
    
    response_data = {
        "message": "🤖 AI Resume Screener API",
        "version": settings.VERSION,
        "status": "active",
        "environment": settings.ENVIRONMENT,
        "timestamp": datetime.utcnow().isoformat(),
        "base_url": base_url,
        "endpoints": {
            "health": f"{base_url}/health",
            "api_health": f"{base_url}/api/health",
            "upload_resume": f"{base_url}/api/upload-resume",
            "score_resume": f"{base_url}/api/score-resume",
            "extract_skills": f"{base_url}/api/extract-skills",
            "analyze_job_description": f"{base_url}/api/analyze-job-description",
            "cleanup": f"{base_url}/api/cleanup"
        }
    }
    
    # Add documentation links for development environment
    if settings.ENVIRONMENT == "development":
        response_data["documentation"] = {
            "swagger_ui": f"{base_url}/docs",
            "redoc": f"{base_url}/redoc",
            "openapi_json": f"{base_url}/openapi.json"
        }
        response_data["quick_start"] = {
            "1_view_docs": f"📚 View API docs: {base_url}/docs",
            "2_test_health": f"🔍 Test health: {base_url}/health",
            "3_upload_resume": f"📄 Upload resume: {base_url}/api/upload-resume",
            "4_score_resume": f"⚡ Score resume: {base_url}/api/score-resume",
            "5_extract_skills": f"🧠 Extract skills: {base_url}/api/extract-skills"
        }
        response_data["testing_urls"] = {
            "postman_collection": f"{base_url}/docs",
            "curl_examples": f"{base_url}/redoc",
            "interactive_testing": f"{base_url}/docs"
        }
    else:
        response_data["documentation"] = "API documentation disabled in production"
    
    return response_data


# Enhanced health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    """
    Comprehensive health check endpoint for monitoring and load balancers.
    
    Returns:
        dict: Application health status and detailed system information
    """
    try:
        # Check spaCy model availability
        import spacy
        spacy_status = "available"
        model_info = {}
        try:
            nlp = spacy.load("en_core_web_sm")
            spacy_status = "loaded"
            model_info = {
                "model_name": "en_core_web_sm",
                "language": "en",
                "pipeline": nlp.pipe_names
            }
        except OSError:
            spacy_status = "model_missing"
    except ImportError:
        spacy_status = "not_installed"
        model_info = {}
    
    # Check directory permissions
    directories_status = {}
    for directory in ["uploads", "logs", "models", "data"]:
        directories_status[directory] = {
            "exists": os.path.exists(directory),
            "writable": os.access(directory, os.W_OK) if os.path.exists(directory) else False
        }
    
    # Check system resources (optional, requires psutil)
    system_info = {}
    try:
        import psutil
        system_info = {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent if os.name != 'nt' else psutil.disk_usage('C:').percent
        }
    except ImportError:
        system_info = {"status": "psutil not installed"}
    
    base_url = get_base_url()
    
    health_data = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "base_url": base_url,
        "services": {
            "spacy": {
                "status": spacy_status,
                "model_info": model_info
            },
            "directories": directories_status,
            "system": system_info
        },
        "api_endpoints": {
            "upload_resume": f"{base_url}/api/upload-resume",
            "score_resume": f"{base_url}/api/score-resume",
            "extract_skills": f"{base_url}/api/extract-skills",
            "analyze_job": f"{base_url}/api/analyze-job-description"
        }
    }
    
    # Add documentation links in development
    if settings.ENVIRONMENT == "development":
        health_data["documentation_links"] = {
            "swagger_ui": f"{base_url}/docs",
            "redoc": f"{base_url}/redoc",
            "openapi_json": f"{base_url}/openapi.json"
        }
        health_data["testing_links"] = {
            "interactive_docs": f"{base_url}/docs",
            "health_check": f"{base_url}/health",
            "api_health": f"{base_url}/api/health"
        }
    
    return health_data


# Include API routes
app.include_router(
    api_router,
    prefix="/api",
    tags=["Resume Screening"]
)


# Enhanced metadata for OpenAPI documentation
if settings.ENVIRONMENT == "development":
    app.openapi_tags = [
        {
            "name": "Health",
            "description": "Health check and system status endpoints",
            "externalDocs": {
                "description": "Health monitoring best practices",
                "url": "https://docs.fastapi.tiangolo.com/advanced/health-checks/"
            }
        },
        {
            "name": "Resume Screening",
            "description": "Resume upload, parsing, scoring, and job matching operations",
            "externalDocs": {
                "description": "Resume screening documentation",
                "url": f"{get_base_url()}/docs"
            }
        }
    ]


# Development server runner with enhanced configuration
if __name__ == "__main__":
    import uvicorn
    
    base_url = get_base_url()
    
    logger.info(f"🚀 Starting development server...")
    logger.info(f"🌐 Server URL: {base_url}")
    logger.info(f"📚 API Documentation: {base_url}/docs")
    logger.info(f"📖 ReDoc Documentation: {base_url}/redoc")
    logger.info(f"🔍 Health Check: {base_url}/health")
    logger.info(f"⚡ API Base URL: {base_url}/api")
    
    # Print all available endpoints
    logger.info(f"📋 Available Endpoints:")
    logger.info(f"   • Root: {base_url}/")
    logger.info(f"   • Health: {base_url}/health")
    logger.info(f"   • Upload Resume: {base_url}/api/upload-resume")
    logger.info(f"   • Score Resume: {base_url}/api/score-resume")
    logger.info(f"   • Extract Skills: {base_url}/api/extract-skills")
    logger.info(f"   • Analyze Job: {base_url}/api/analyze-job-description")
    logger.info(f"   • Cleanup: {base_url}/api/cleanup")
    
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.ENVIRONMENT == "development",
        log_level=settings.LOG_LEVEL.lower(),
        access_log=True,
        reload_dirs=["app"] if settings.ENVIRONMENT == "development" else None,
        reload_includes=["*.py"] if settings.ENVIRONMENT == "development" else None
    )
