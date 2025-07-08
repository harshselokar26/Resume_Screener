"""
Core Package for AI Resume Screener

This package contains core functionality including security, authentication,
authorization, and other fundamental components of the application.
"""

from app.core.security import (
    # Password utilities
    hash_password,
    verify_password,
    generate_password_hash,
    
    # Token utilities
    create_access_token,
    create_refresh_token,
    verify_token,
    decode_token,
    
    # API Key utilities
    generate_api_key,
    verify_api_key,
    hash_api_key,
    
    # Security utilities
    generate_secure_random_string,
    create_csrf_token,
    verify_csrf_token,
    sanitize_input,
    rate_limit_key,
    
    # Authentication dependencies
    get_current_user,
    get_current_active_user,
    require_api_key,
    require_admin_role,
    
    # Security classes
    SecurityManager,
    TokenManager,
    RateLimiter
)

__all__ = [
    # Password utilities
    "hash_password",
    "verify_password", 
    "generate_password_hash",
    
    # Token utilities
    "create_access_token",
    "create_refresh_token",
    "verify_token",
    "decode_token",
    
    # API Key utilities
    "generate_api_key",
    "verify_api_key",
    "hash_api_key",
    
    # Security utilities
    "generate_secure_random_string",
    "create_csrf_token",
    "verify_csrf_token",
    "sanitize_input",
    "rate_limit_key",
    
    # Authentication dependencies
    "get_current_user",
    "get_current_active_user",
    "require_api_key",
    "require_admin_role",
    
    # Security classes
    "SecurityManager",
    "TokenManager",
    "RateLimiter"
]
