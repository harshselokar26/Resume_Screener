"""
Application Settings and Configuration Management

This module handles all application configuration using Pydantic Settings
for environment variable management and validation.
"""

import os
from datetime import datetime
from typing import List, Optional, Union
from pydantic import BaseSettings, Field, validator
from functools import lru_cache
import secrets


class Settings(BaseSettings):
    """
    Application settings with environment variable support.
    
    All settings can be overridden by environment variables with the same name.
    Example: PROJECT_NAME environment variable overrides project_name setting.
    """
    
    # ===== APPLICATION SETTINGS =====
    PROJECT_NAME: str = Field(
        default="AI Resume Screener",
        description="Name of the application"
    )
    
    PROJECT_DESCRIPTION: str = Field(
        default="AI-powered resume screening and job matching system using NLP",
        description="Application description"
    )
    
    VERSION: str = Field(
        default="1.0.0",
        description="Application version"
    )
    
    ENVIRONMENT: str = Field(
        default="development",
        description="Application environment (development, staging, production)"
    )
    
    DEBUG: bool = Field(
        default=True,
        description="Enable debug mode"
    )
    
    # ===== SERVER SETTINGS =====
    HOST: str = Field(
        default="0.0.0.0",
        description="Server host address"
    )
    
    PORT: int = Field(
        default=8000,
        description="Server port number"
    )
    
    WORKERS: int = Field(
        default=1,
        description="Number of worker processes"
    )
    
    # ===== SECURITY SETTINGS =====
    SECRET_KEY: str = Field(
        default_factory=lambda: secrets.token_urlsafe(32),
        description="Secret key for encryption and signing"
    )
    
    ALLOWED_HOSTS: List[str] = Field(
        default=["*"],
        description="List of allowed host headers"
    )
    
    ALLOWED_ORIGINS: List[str] = Field(
        default=["*"],
        description="List of allowed CORS origins"
    )
    
    # ===== FILE HANDLING SETTINGS =====
    MAX_FILE_SIZE: int = Field(
        default=10 * 1024 * 1024,  # 10MB
        description="Maximum file upload size in bytes"
    )
    
    ALLOWED_FILE_TYPES: List[str] = Field(
        default=["pdf", "doc", "docx"],
        description="Allowed file extensions for resume uploads"
    )
    
    UPLOAD_DIR: str = Field(
        default="uploads",
        description="Directory for temporary file uploads"
    )
    
    # ===== NLP SETTINGS =====
    SPACY_MODEL: str = Field(
        default="en_core_web_sm",
        description="spaCy model name for NLP processing"
    )
    
    MIN_SIMILARITY_SCORE: float = Field(
        default=0.1,
        description="Minimum similarity score threshold"
    )
    
    MAX_SIMILARITY_SCORE: float = Field(
        default=1.0,
        description="Maximum similarity score threshold"
    )
    
    # ===== LOGGING SETTINGS =====
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    
    LOG_FILE: str = Field(
        default="logs/app.log",
        description="Log file path"
    )
    
    LOG_MAX_SIZE: int = Field(
        default=10 * 1024 * 1024,  # 10MB
        description="Maximum log file size in bytes"
    )
    
    LOG_BACKUP_COUNT: int = Field(
        default=5,
        description="Number of log backup files to keep"
    )
    
    # ===== DATABASE SETTINGS (Optional) =====
    DATABASE_URL: Optional[str] = Field(
        default=None,
        description="Database connection URL (if using database)"
    )
    
    # ===== REDIS SETTINGS (Optional) =====
    REDIS_URL: Optional[str] = Field(
        default=None,
        description="Redis connection URL (if using Redis for caching)"
    )
    
    # ===== API SETTINGS =====
    API_PREFIX: str = Field(
        default="/api",
        description="API route prefix"
    )
    
    API_RATE_LIMIT: int = Field(
        default=100,
        description="API rate limit per minute"
    )
    
    # ===== MONITORING SETTINGS =====
    ENABLE_METRICS: bool = Field(
        default=True,
        description="Enable application metrics collection"
    )
    
    HEALTH_CHECK_INTERVAL: int = Field(
        default=30,
        description="Health check interval in seconds"
    )
    
    class Config:
        """Pydantic configuration"""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        
    @validator("ENVIRONMENT")
    def validate_environment(cls, v):
        """Validate environment setting"""
        allowed_envs = ["development", "staging", "production"]
        if v.lower() not in allowed_envs:
            raise ValueError(f"Environment must be one of: {allowed_envs}")
        return v.lower()
    
    @validator("LOG_LEVEL")
    def validate_log_level(cls, v):
        """Validate log level setting"""
        allowed_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in allowed_levels:
            raise ValueError(f"Log level must be one of: {allowed_levels}")
        return v.upper()
    
    @validator("ALLOWED_HOSTS", pre=True)
    def parse_allowed_hosts(cls, v):
        """Parse allowed hosts from string or list"""
        if isinstance(v, str):
            return [host.strip() for host in v.split(",")]
        return v
    
    @validator("ALLOWED_ORIGINS", pre=True)
    def parse_allowed_origins(cls, v):
        """Parse allowed origins from string or list"""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    @validator("ALLOWED_FILE_TYPES", pre=True)
    def parse_allowed_file_types(cls, v):
        """Parse allowed file types from string or list"""
        if isinstance(v, str):
            return [ext.strip().lower() for ext in v.split(",")]
        return [ext.lower() for ext in v]
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.ENVIRONMENT == "development"
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.ENVIRONMENT == "production"
    
    @property
    def is_staging(self) -> bool:
        """Check if running in staging environment"""
        return self.ENVIRONMENT == "staging"
    
    def get_current_timestamp(self) -> str:
        """Get current timestamp in ISO format"""
        return datetime.utcnow().isoformat()
    
    def get_upload_path(self, filename: str) -> str:
        """Get full upload path for a file"""
        return os.path.join(self.UPLOAD_DIR, filename)
    
    def is_allowed_file_type(self, filename: str) -> bool:
        """Check if file type is allowed"""
        if not filename:
            return False
        extension = filename.split(".")[-1].lower()
        return extension in self.ALLOWED_FILE_TYPES
    
    def get_file_size_mb(self, size_bytes: int) -> float:
        """Convert bytes to MB"""
        return size_bytes / (1024 * 1024)
    
    def is_file_size_allowed(self, size_bytes: int) -> bool:
        """Check if file size is within allowed limits"""
        return size_bytes <= self.MAX_FILE_SIZE


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Using lru_cache ensures settings are loaded only once and cached
    for subsequent calls, improving performance.
    """
    return Settings()


# Global settings instance
settings = get_settings()
