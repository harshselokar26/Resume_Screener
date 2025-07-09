"""
Utilities Package for AI Resume Screener

This package contains utility functions and classes used throughout the application.
"""

from app.utils.exceptions import (
    # Base exception
    ResumeScreenerException,
    
    # Validation exceptions
    ValidationError,
    InvalidInputError,
    MissingRequiredFieldError,
    InvalidRangeError,
    
    # File processing exceptions
    FileProcessingError,
    FileValidationError,
    FileNotFoundError,
    FileSizeExceededError,
    UnsupportedFileTypeError,
    FileCorruptedError,
    
    # NLP processing exceptions
    NLPProcessingError,
    ModelNotLoadedError,
    TextProcessingError,
    SkillExtractionError,
    
    # Scoring exceptions
    ScoringError,
    SimilarityCalculationError,
    InsufficientDataError,
    
    # Authentication exceptions
    AuthenticationError,
    AuthorizationError,
    TokenExpiredError,
    InvalidTokenError,
    
    # Database exceptions
    DatabaseError,
    DatabaseConnectionError,
    RecordNotFoundError,
    DuplicateRecordError,
    
    # External service exceptions
    ExternalServiceError,
    ServiceUnavailableError,
    APIRateLimitError,
    
    # Configuration exceptions
    ConfigurationError,
    MissingConfigurationError,
    InvalidConfigurationError,
    
    # Utility functions
    handle_exception,
    create_error_response,
)

__all__ = [
    # Base exception
    "ResumeScreenerException",
    
    # Validation exceptions
    "ValidationError",
    "InvalidInputError",
    "MissingRequiredFieldError",
    "InvalidRangeError",
    
    # File processing exceptions
    "FileProcessingError",
    "FileValidationError",
    "FileNotFoundError",
    "FileSizeExceededError",
    "UnsupportedFileTypeError",
    "FileCorruptedError",
    
    # NLP processing exceptions
    "NLPProcessingError",
    "ModelNotLoadedError",
    "TextProcessingError",
    "SkillExtractionError",
    
    # Scoring exceptions
    "ScoringError",
    "SimilarityCalculationError",
    "InsufficientDataError",
    
    # Authentication exceptions
    "AuthenticationError",
    "AuthorizationError",
    "TokenExpiredError",
    "InvalidTokenError",
    
    # Database exceptions
    "DatabaseError",
    "DatabaseConnectionError",
    "RecordNotFoundError",
    "DuplicateRecordError",
    
    # External service exceptions
    "ExternalServiceError",
    "ServiceUnavailableError",
    "APIRateLimitError",
    
    # Configuration exceptions
    "ConfigurationError",
    "MissingConfigurationError",
    "InvalidConfigurationError",
    
    # Utility functions
    "handle_exception",
    "create_error_response",
]
