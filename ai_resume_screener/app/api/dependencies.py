"""
FastAPI Dependencies for AI Resume Screener

This module contains dependency functions used across API endpoints
for authentication, validation, and service injection.
"""

import os
from typing import Optional, Generator
from fastapi import Depends, HTTPException, UploadFile, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import logging

from app.config.settings import settings
from app.services.file_handler import FileHandler
from app.utils.exceptions import FileValidationError
from app.utils.validators import validate_file_type, validate_file_size

# Setup logging
logger = logging.getLogger(__name__)

# Security scheme for authentication
security = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[str]:
    """
    Get current authenticated user (placeholder for future authentication).
    
    Args:
        credentials: HTTP Bearer token credentials
        
    Returns:
        Optional[str]: User identifier or None if not authenticated
        
    Note:
        This is a placeholder for future authentication implementation.
        Currently returns None (no authentication required).
    """
    if not credentials:
        return None
    
    # TODO: Implement actual token validation
    # For now, return a placeholder user
    return "anonymous_user"


async def validate_file_upload(file: UploadFile) -> None:
    """
    Validate uploaded file for type, size, and content.
    
    Args:
        file: Uploaded file to validate
        
    Raises:
        HTTPException: If file validation fails
    """
    try:
        # Check if file is provided
        if not file or not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No file provided"
            )
        
        # Validate file type
        if not validate_file_type(file.filename):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File type not allowed. Allowed types: {settings.ALLOWED_FILE_TYPES}"
            )
        
        # Read file content to check size
        content = await file.read()
        file_size = len(content)
        
        # Reset file pointer
        await file.seek(0)
        
        # Validate file size
        if not validate_file_size(file_size):
            max_size_mb = settings.get_file_size_mb(settings.MAX_FILE_SIZE)
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large. Maximum size: {max_size_mb:.1f}MB"
            )
        
        # Check if file is empty
        if file_size == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File is empty"
            )
        
        # Basic content validation (check for PDF/DOC headers)
        if not _validate_file_content(content, file.filename):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File content does not match file extension"
            )
        
        logger.info(f"File validation passed: {file.filename} ({file_size} bytes)")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="File validation failed"
        )


def _validate_file_content(content: bytes, filename: str) -> bool:
    """
    Validate file content matches the file extension.
    
    Args:
        content: File content bytes
        filename: Original filename
        
    Returns:
        bool: True if content matches extension
    """
    if not content:
        return False
    
    extension = filename.lower().split('.')[-1]
    
    # PDF file validation
    if extension == 'pdf':
        return content.startswith(b'%PDF-')
    
    # DOC file validation
    elif extension == 'doc':
        return (
            content.startswith(b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1') or  # OLE header
            content.startswith(b'\x50\x4b\x03\x04')  # ZIP header (some DOC files)
        )
    
    # DOCX file validation
    elif extension == 'docx':
        return content.startswith(b'\x50\x4b\x03\x04')  # ZIP header
    
    return True


def get_file_handler() -> FileHandler:
    """
    Get file handler service instance.
    
    Returns:
        FileHandler: File handling service
    """
    return FileHandler()


def get_upload_directory() -> str:
    """
    Get upload directory path, creating it if necessary.
    
    Returns:
        str: Upload directory path
    """
    upload_dir = settings.UPLOAD_DIR
    os.makedirs(upload_dir, exist_ok=True)
    return upload_dir


def validate_text_input(text: str, min_length: int = 10, max_length: int = 50000) -> str:
    """
    Validate text input for API endpoints.
    
    Args:
        text: Input text to validate
        min_length: Minimum text length
        max_length: Maximum text length
        
    Returns:
        str: Validated and cleaned text
        
    Raises:
        HTTPException: If text validation fails
    """
    if not text or not text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Text input cannot be empty"
        )
    
    text = text.strip()
    
    if len(text) < min_length:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Text too short. Minimum length: {min_length} characters"
        )
    
    if len(text) > max_length:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Text too long. Maximum length: {max_length} characters"
        )
    
    return text


def check_service_health() -> bool:
    """
    Check if all required services are healthy.
    
    Returns:
        bool: True if all services are healthy
    """
    try:
        # Check upload directory
        if not os.path.exists(settings.UPLOAD_DIR):
            return False
        
        if not os.access(settings.UPLOAD_DIR, os.W_OK):
            return False
        
        # Check spaCy model
        try:
            import spacy
            spacy.load(settings.SPACY_MODEL)
        except (ImportError, OSError):
            return False
        
        return True
        
    except Exception:
        return False


def get_request_id() -> str:
    """
    Generate unique request ID for tracking.
    
    Returns:
        str: Unique request identifier
    """
    import uuid
    return str(uuid.uuid4())


class CommonQueryParams:
    """
    Common query parameters for API endpoints.
    """
    
    def __init__(
        self,
        skip: int = 0,
        limit: int = 100,
        include_metadata: bool = False
    ):
        self.skip = skip
        self.limit = limit
        self.include_metadata = include_metadata


def get_common_params(
    skip: int = 0,
    limit: int = 100,
    include_metadata: bool = False
) -> CommonQueryParams:
    """
    Get common query parameters dependency.
    
    Args:
        skip: Number of items to skip
        limit: Maximum number of items to return
        include_metadata: Whether to include metadata in response
        
    Returns:
        CommonQueryParams: Common parameters object
    """
    return CommonQueryParams(skip, limit, include_metadata)
