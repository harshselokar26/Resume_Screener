"""
Utils Package for AI Resume Screener

This package contains utility functions, validators, and custom exceptions
used throughout the application for common operations and error handling.
"""

from app.utils.helpers import (
    generate_unique_filename,
    cleanup_temp_files,
    format_file_size,
    calculate_similarity_percentage,
    extract_email_from_text,
    extract_phone_from_text,
    sanitize_filename,
    get_file_extension,
    is_valid_uuid,
    truncate_text,
    normalize_skill_name,
    calculate_reading_time,
    get_current_timestamp,
    hash_text,
    mask_sensitive_data
)

from app.utils.validators import (
    validate_file_type,
    validate_file_size,
    validate_email,
    validate_phone,
    validate_url,
    validate_text_length,
    validate_similarity_score,
    validate_percentage,
    validate_uuid,
    validate_json_structure,
    validate_skill_name,
    validate_date_format
)

from app.utils.exceptions import (
    ResumeScreenerException,
    FileProcessingError,
    NLPProcessingError,
    ScoringError,
    ValidationError,
    AuthenticationError,
    AuthorizationError,
    DatabaseError,
    ExternalServiceError,
    ConfigurationError
)

__all__ = [
    # Helper functions
    "generate_unique_filename",
    "cleanup_temp_files",
    "format_file_size",
    "calculate_similarity_percentage",
    "extract_email_from_text",
    "extract_phone_from_text",
    "sanitize_filename",
    "get_file_extension",
    "is_valid_uuid",
    "truncate_text",
    "normalize_skill_name",
    "calculate_reading_time",
    "get_current_timestamp",
    "hash_text",
    "mask_sensitive_data",
    
    # Validators
    "validate_file_type",
    "validate_file_size",
    "validate_email",
    "validate_phone",
    "validate_url",
    "validate_text_length",
    "validate_similarity_score",
    "validate_percentage",
    "validate_uuid",
    "validate_json_structure",
    "validate_skill_name",
    "validate_date_format",
    
    # Exceptions
    "ResumeScreenerException",
    "FileProcessingError",
    "NLPProcessingError",
    "ScoringError",
    "ValidationError",
    "AuthenticationError",
    "AuthorizationError",
    "DatabaseError",
    "ExternalServiceError",
    "ConfigurationError"
]
