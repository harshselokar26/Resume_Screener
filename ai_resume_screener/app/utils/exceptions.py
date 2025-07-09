"""
Custom Exception Classes for AI Resume Screener

This module defines custom exception classes for different types of errors
that can occur in the application, providing structured error handling.
"""

from datetime import datetime
from typing import Optional, Dict, Any


class ResumeScreenerException(Exception):
    """
    Base exception class for all Resume Screener related errors.
    """
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize base exception.
        
        Args:
            message: Error message
            error_code: Specific error code
            status_code: HTTP status code
            details: Additional error details
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.status_code = status_code
        self.details = details or {}
        self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert exception to dictionary.
        
        Returns:
            Dict containing exception details
        """
        return {
            "error": True,
            "message": self.message,
            "error_code": self.error_code,
            "status_code": self.status_code,
            "details": self.details,
            "timestamp": self.timestamp.isoformat()
        }
    
    def __str__(self) -> str:
        """String representation of exception."""
        return f"{self.error_code}: {self.message}"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"{self.__class__.__name__}(message='{self.message}', error_code='{self.error_code}', status_code={self.status_code})"


# ===== VALIDATION EXCEPTIONS (MUST BE DEFINED FIRST) =====

class ValidationError(ResumeScreenerException):
    """Exception raised when input validation fails."""
    
    def __init__(
        self,
        message: str,
        field_name: Optional[str] = None,
        field_value: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, status_code=400, **kwargs)
        self.field_name = field_name
        self.field_value = field_value
        if field_name:
            self.details["field_name"] = field_name
        if field_value:
            self.details["field_value"] = str(field_value)[:100]  # Limit length


class FileValidationError(ValidationError):
    """Exception raised when file validation fails."""
    
    def __init__(
        self,
        message: str,
        filename: Optional[str] = None,
        file_type: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.filename = filename
        self.file_type = file_type
        if filename:
            self.details["filename"] = filename
        if file_type:
            self.details["file_type"] = file_type


class InvalidInputError(ValidationError):
    """Exception raised when input format is invalid."""
    
    def __init__(self, field_name: str, expected_format: str, **kwargs):
        super().__init__(
            f"Invalid input format for {field_name}. Expected: {expected_format}",
            field_name=field_name,
            error_code="INVALID_INPUT_FORMAT",
            **kwargs
        )
        self.details["expected_format"] = expected_format


class MissingRequiredFieldError(ValidationError):
    """Exception raised when required field is missing."""
    
    def __init__(self, field_name: str, **kwargs):
        super().__init__(
            f"Missing required field: {field_name}",
            field_name=field_name,
            error_code="MISSING_REQUIRED_FIELD",
            **kwargs
        )


class InvalidRangeError(ValidationError):
    """Exception raised when value is outside valid range."""
    
    def __init__(self, field_name: str, value: Any, min_value: Any, max_value: Any, **kwargs):
        super().__init__(
            f"Value for {field_name} ({value}) is outside valid range [{min_value}, {max_value}]",
            field_name=field_name,
            field_value=str(value),
            error_code="INVALID_RANGE",
            **kwargs
        )
        self.details.update({
            "value": value,
            "min_value": min_value,
            "max_value": max_value
        })


# ===== FILE PROCESSING EXCEPTIONS =====

class FileProcessingError(ResumeScreenerException):
    """Exception raised when file processing fails."""
    
    def __init__(
        self,
        message: str,
        filename: Optional[str] = None,
        file_type: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, status_code=422, **kwargs)
        self.filename = filename
        self.file_type = file_type
        if filename:
            self.details["filename"] = filename
        if file_type:
            self.details["file_type"] = file_type


class FileNotFoundError(FileProcessingError):
    """Exception raised when file is not found."""
    
    def __init__(self, filename: str, **kwargs):
        super().__init__(
            f"File not found: {filename}",
            filename=filename,
            error_code="FILE_NOT_FOUND",
            status_code=404,
            **kwargs
        )


class FileSizeExceededError(FileProcessingError):
    """Exception raised when file size exceeds limit."""
    
    def __init__(self, file_size: int, max_size: int, **kwargs):
        super().__init__(
            f"File size ({file_size} bytes) exceeds maximum allowed size ({max_size} bytes)",
            error_code="FILE_SIZE_EXCEEDED",
            status_code=413,
            **kwargs
        )
        self.details.update({
            "file_size": file_size,
            "max_size": max_size
        })


class UnsupportedFileTypeError(FileProcessingError):
    """Exception raised when file type is not supported."""
    
    def __init__(self, file_type: str, supported_types: list, **kwargs):
        super().__init__(
            f"Unsupported file type: {file_type}. Supported types: {', '.join(supported_types)}",
            file_type=file_type,
            error_code="UNSUPPORTED_FILE_TYPE",
            status_code=415,
            **kwargs
        )
        self.details.update({
            "file_type": file_type,
            "supported_types": supported_types
        })


# ===== NLP PROCESSING EXCEPTIONS =====

class NLPProcessingError(ResumeScreenerException):
    """Exception raised when NLP processing fails."""
    
    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        text_length: Optional[int] = None,
        **kwargs
    ):
        super().__init__(message, status_code=422, **kwargs)
        self.operation = operation
        self.text_length = text_length
        if operation:
            self.details["operation"] = operation
        if text_length:
            self.details["text_length"] = text_length


class ModelNotLoadedError(NLPProcessingError):
    """Exception raised when NLP model is not loaded."""
    
    def __init__(self, model_name: str, **kwargs):
        super().__init__(
            f"NLP model not loaded: {model_name}",
            error_code="MODEL_NOT_LOADED",
            status_code=503,
            **kwargs
        )
        self.details["model_name"] = model_name


class SkillExtractionError(NLPProcessingError):
    """Exception raised when skill extraction fails."""
    
    def __init__(self, **kwargs):
        super().__init__(
            "Failed to extract skills from text",
            operation="skill_extraction",
            error_code="SKILL_EXTRACTION_ERROR",
            **kwargs
        )


# ===== SCORING EXCEPTIONS =====

class ScoringError(ResumeScreenerException):
    """Exception raised when resume scoring fails."""
    
    def __init__(
        self,
        message: str,
        resume_length: Optional[int] = None,
        job_description_length: Optional[int] = None,
        **kwargs
    ):
        super().__init__(message, status_code=422, **kwargs)
        if resume_length:
            self.details["resume_length"] = resume_length
        if job_description_length:
            self.details["job_description_length"] = job_description_length


class SimilarityCalculationError(ScoringError):
    """Exception raised when similarity calculation fails."""
    
    def __init__(self, **kwargs):
        super().__init__(
            "Failed to calculate similarity score",
            error_code="SIMILARITY_CALCULATION_ERROR",
            **kwargs
        )


# ===== AUTHENTICATION EXCEPTIONS =====

class AuthenticationError(ResumeScreenerException):
    """Exception raised when authentication fails."""
    
    def __init__(self, message: str = "Authentication failed", **kwargs):
        super().__init__(
            message,
            error_code="AUTHENTICATION_FAILED",
            status_code=401,
            **kwargs
        )


class AuthorizationError(ResumeScreenerException):
    """Exception raised when authorization fails."""
    
    def __init__(self, message: str = "Access denied", **kwargs):
        super().__init__(
            message,
            error_code="ACCESS_DENIED",
            status_code=403,
            **kwargs
        )


# ===== DATABASE EXCEPTIONS =====

class DatabaseError(ResumeScreenerException):
    """Exception raised when database operations fail."""
    
    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, status_code=500, **kwargs)
        if operation:
            self.details["operation"] = operation


# ===== CONFIGURATION EXCEPTIONS =====

class ConfigurationError(ResumeScreenerException):
    """Exception raised when configuration is invalid."""
    
    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            message,
            error_code="CONFIGURATION_ERROR",
            status_code=500,
            **kwargs
        )
        if config_key:
            self.details["config_key"] = config_key


# ===== EXTERNAL SERVICE EXCEPTIONS =====

class ExternalServiceError(ResumeScreenerException):
    """Exception raised when external service calls fail."""
    
    def __init__(
        self,
        message: str,
        service_name: Optional[str] = None,
        status_code: int = 502,
        **kwargs
    ):
        super().__init__(message, status_code=status_code, **kwargs)
        if service_name:
            self.details["service_name"] = service_name
