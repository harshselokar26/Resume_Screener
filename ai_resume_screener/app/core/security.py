"""
Security Module for AI Resume Screener

This module handles authentication, authorization, password hashing,
token management, and other security-related functionality.
"""

import secrets
import hashlib
import hmac
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List, Union
import re
import time
from functools import wraps

# Cryptography and JWT
try:
    import bcrypt
    BCRYPT_AVAILABLE = True
except ImportError:
    BCRYPT_AVAILABLE = False

try:
    import jwt
    from jwt.exceptions import InvalidTokenError, ExpiredSignatureError
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False

# FastAPI dependencies
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader

from app.config.settings import settings
from app.utils.exceptions import (
    AuthenticationError,
    AuthorizationError,
    TokenExpiredError,
    InvalidTokenError as CustomInvalidTokenError
)

# Setup logging
logger = logging.getLogger(__name__)

# Security schemes
security_bearer = HTTPBearer(auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


# ===== PASSWORD UTILITIES =====

def hash_password(password: str) -> str:
    """
    Hash password using bcrypt.
    
    Args:
        password: Plain text password
        
    Returns:
        str: Hashed password
        
    Raises:
        RuntimeError: If bcrypt is not available
    """
    if not BCRYPT_AVAILABLE:
        # Fallback to PBKDF2 if bcrypt is not available
        return _hash_password_pbkdf2(password)
    
    try:
        # Generate salt and hash password
        salt = bcrypt.gensalt(rounds=12)
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    except Exception as e:
        logger.error(f"Password hashing failed: {str(e)}")
        raise RuntimeError("Password hashing failed")


def verify_password(password: str, hashed_password: str) -> bool:
    """
    Verify password against hash.
    
    Args:
        password: Plain text password
        hashed_password: Hashed password
        
    Returns:
        bool: True if password matches
    """
    if not password or not hashed_password:
        return False
    
    try:
        if BCRYPT_AVAILABLE and hashed_password.startswith('$2b$'):
            # bcrypt hash
            return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))
        else:
            # PBKDF2 hash (fallback)
            return _verify_password_pbkdf2(password, hashed_password)
    except Exception as e:
        logger.error(f"Password verification failed: {str(e)}")
        return False


def _hash_password_pbkdf2(password: str) -> str:
    """Fallback password hashing using PBKDF2."""
    salt = secrets.token_hex(32)
    pwdhash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 100000)
    return f"pbkdf2_sha256${salt}${pwdhash.hex()}"


def _verify_password_pbkdf2(password: str, hashed_password: str) -> bool:
    """Fallback password verification using PBKDF2."""
    try:
        algorithm, salt, stored_hash = hashed_password.split('$')
        if algorithm != 'pbkdf2_sha256':
            return False
        
        pwdhash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 100000)
        return hmac.compare_digest(stored_hash, pwdhash.hex())
    except ValueError:
        return False


def generate_password_hash(password: str) -> str:
    """
    Generate password hash (alias for hash_password).
    
    Args:
        password: Plain text password
        
    Returns:
        str: Hashed password
    """
    return hash_password(password)


def validate_password_strength(password: str) -> Dict[str, Any]:
    """
    Validate password strength.
    
    Args:
        password: Password to validate
        
    Returns:
        Dict containing validation results
    """
    if not password:
        return {"valid": False, "errors": ["Password cannot be empty"]}
    
    errors = []
    score = 0
    
    # Length check
    if len(password) < 8:
        errors.append("Password must be at least 8 characters long")
    else:
        score += 1
    
    # Uppercase check
    if not re.search(r'[A-Z]', password):
        errors.append("Password must contain at least one uppercase letter")
    else:
        score += 1
    
    # Lowercase check
    if not re.search(r'[a-z]', password):
        errors.append("Password must contain at least one lowercase letter")
    else:
        score += 1
    
    # Number check
    if not re.search(r'\d', password):
        errors.append("Password must contain at least one number")
    else:
        score += 1
    
    # Special character check
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        errors.append("Password must contain at least one special character")
    else:
        score += 1
    
    # Common password check
    common_passwords = ['password', '123456', 'qwerty', 'abc123', 'password123']
    if password.lower() in common_passwords:
        errors.append("Password is too common")
        score -= 1
    
    strength_levels = {
        0: "Very Weak",
        1: "Weak", 
        2: "Fair",
        3: "Good",
        4: "Strong",
        5: "Very Strong"
    }
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "score": max(0, score),
        "strength": strength_levels.get(max(0, score), "Unknown")
    }


# ===== TOKEN UTILITIES =====

class TokenManager:
    """Token management class for JWT operations."""
    
    def __init__(self):
        """Initialize token manager."""
        self.secret_key = settings.SECRET_KEY
        self.algorithm = "HS256"
        self.access_token_expire_minutes = 30
        self.refresh_token_expire_days = 7
    
    def create_access_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """
        Create JWT access token.
        
        Args:
            data: Data to encode in token
            expires_delta: Token expiration time
            
        Returns:
            str: JWT token
        """
        if not JWT_AVAILABLE:
            raise RuntimeError("JWT library not available")
        
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(minutes=self.access_token_expire_minutes)
        
        to_encode.update({
            "exp": expire,
            "iat": datetime.now(timezone.utc),
            "type": "access"
        })
        
        try:
            encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
            return encoded_jwt
        except Exception as e:
            logger.error(f"Token creation failed: {str(e)}")
            raise AuthenticationError("Failed to create access token")
    
    def create_refresh_token(self, data: Dict[str, Any]) -> str:
        """
        Create JWT refresh token.
        
        Args:
            data: Data to encode in token
            
        Returns:
            str: JWT refresh token
        """
        if not JWT_AVAILABLE:
            raise RuntimeError("JWT library not available")
        
        to_encode = data.copy()
        expire = datetime.now(timezone.utc) + timedelta(days=self.refresh_token_expire_days)
        
        to_encode.update({
            "exp": expire,
            "iat": datetime.now(timezone.utc),
            "type": "refresh"
        })
        
        try:
            encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
            return encoded_jwt
        except Exception as e:
            logger.error(f"Refresh token creation failed: {str(e)}")
            raise AuthenticationError("Failed to create refresh token")
    
    def verify_token(self, token: str, token_type: str = "access") -> Dict[str, Any]:
        """
        Verify and decode JWT token.
        
        Args:
            token: JWT token to verify
            token_type: Expected token type (access/refresh)
            
        Returns:
            Dict: Decoded token payload
            
        Raises:
            AuthenticationError: If token is invalid
        """
        if not JWT_AVAILABLE:
            raise RuntimeError("JWT library not available")
        
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Verify token type
            if payload.get("type") != token_type:
                raise CustomInvalidTokenError("Invalid token type")
            
            return payload
            
        except ExpiredSignatureError:
            raise TokenExpiredError("Token has expired")
        except InvalidTokenError as e:
            raise CustomInvalidTokenError(f"Invalid token: {str(e)}")
        except Exception as e:
            logger.error(f"Token verification failed: {str(e)}")
            raise AuthenticationError("Token verification failed")
    
    def decode_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Decode token without verification (for debugging).
        
        Args:
            token: JWT token to decode
            
        Returns:
            Optional[Dict]: Decoded payload or None
        """
        if not JWT_AVAILABLE:
            return None
        
        try:
            # Decode without verification
            payload = jwt.decode(token, options={"verify_signature": False})
            return payload
        except Exception:
            return None


# Global token manager instance
token_manager = TokenManager()

# Convenience functions
def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create access token."""
    return token_manager.create_access_token(data, expires_delta)

def create_refresh_token(data: Dict[str, Any]) -> str:
    """Create refresh token."""
    return token_manager.create_refresh_token(data)

def verify_token(token: str, token_type: str = "access") -> Dict[str, Any]:
    """Verify token."""
    return token_manager.verify_token(token, token_type)

def decode_token(token: str) -> Optional[Dict[str, Any]]:
    """Decode token."""
    return token_manager.decode_token(token)


# ===== API KEY UTILITIES =====

def generate_api_key(prefix: str = "rsc", length: int = 32) -> str:
    """
    Generate secure API key.
    
    Args:
        prefix: Key prefix
        length: Key length
        
    Returns:
        str: Generated API key
    """
    random_part = secrets.token_urlsafe(length)
    return f"{prefix}_{random_part}"


def hash_api_key(api_key: str) -> str:
    """
    Hash API key for storage.
    
    Args:
        api_key: API key to hash
        
    Returns:
        str: Hashed API key
    """
    return hashlib.sha256(api_key.encode()).hexdigest()


def verify_api_key(api_key: str, hashed_key: str) -> bool:
    """
    Verify API key against hash.
    
    Args:
        api_key: API key to verify
        hashed_key: Stored hash
        
    Returns:
        bool: True if key matches
    """
    if not api_key or not hashed_key:
        return False
    
    return hmac.compare_digest(hash_api_key(api_key), hashed_key)


# ===== SECURITY UTILITIES =====

def generate_secure_random_string(length: int = 32) -> str:
    """
    Generate cryptographically secure random string.
    
    Args:
        length: String length
        
    Returns:
        str: Random string
    """
    return secrets.token_urlsafe(length)


def create_csrf_token(session_id: str) -> str:
    """
    Create CSRF token.
    
    Args:
        session_id: Session identifier
        
    Returns:
        str: CSRF token
    """
    timestamp = str(int(time.time()))
    data = f"{session_id}:{timestamp}"
    signature = hmac.new(
        settings.SECRET_KEY.encode(),
        data.encode(),
        hashlib.sha256
    ).hexdigest()
    
    return f"{timestamp}:{signature}"


def verify_csrf_token(token: str, session_id: str, max_age: int = 3600) -> bool:
    """
    Verify CSRF token.
    
    Args:
        token: CSRF token to verify
        session_id: Session identifier
        max_age: Maximum token age in seconds
        
    Returns:
        bool: True if token is valid
    """
    try:
        timestamp_str, signature = token.split(':', 1)
        timestamp = int(timestamp_str)
        
        # Check token age
        if time.time() - timestamp > max_age:
            return False
        
        # Verify signature
        data = f"{session_id}:{timestamp_str}"
        expected_signature = hmac.new(
            settings.SECRET_KEY.encode(),
            data.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return hmac.compare_digest(signature, expected_signature)
        
    except (ValueError, TypeError):
        return False


def sanitize_input(input_string: str, max_length: int = 1000) -> str:
    """
    Sanitize user input.
    
    Args:
        input_string: Input to sanitize
        max_length: Maximum allowed length
        
    Returns:
        str: Sanitized input
    """
    if not isinstance(input_string, str):
        return ""
    
    # Truncate if too long
    if len(input_string) > max_length:
        input_string = input_string[:max_length]
    
    # Remove potentially dangerous characters
    sanitized = re.sub(r'[<>"\']', '', input_string)
    
    # Remove control characters
    sanitized = ''.join(char for char in sanitized if ord(char) >= 32)
    
    return sanitized.strip()


def rate_limit_key(identifier: str, endpoint: str) -> str:
    """
    Generate rate limit key.
    
    Args:
        identifier: User/IP identifier
        endpoint: API endpoint
        
    Returns:
        str: Rate limit key
    """
    return f"rate_limit:{identifier}:{endpoint}"


# ===== RATE LIMITING =====

class RateLimiter:
    """Simple in-memory rate limiter."""
    
    def __init__(self):
        """Initialize rate limiter."""
        self.requests = {}
        self.cleanup_interval = 300  # 5 minutes
        self.last_cleanup = time.time()
    
    def is_allowed(self, key: str, limit: int, window: int) -> bool:
        """
        Check if request is allowed.
        
        Args:
            key: Rate limit key
            limit: Request limit
            window: Time window in seconds
            
        Returns:
            bool: True if request is allowed
        """
        current_time = time.time()
        
        # Cleanup old entries periodically
        if current_time - self.last_cleanup > self.cleanup_interval:
            self._cleanup(current_time)
        
        # Get or create request history for key
        if key not in self.requests:
            self.requests[key] = []
        
        request_times = self.requests[key]
        
        # Remove old requests outside the window
        cutoff_time = current_time - window
        request_times[:] = [t for t in request_times if t > cutoff_time]
        
        # Check if limit is exceeded
        if len(request_times) >= limit:
            return False
        
        # Add current request
        request_times.append(current_time)
        return True
    
    def _cleanup(self, current_time: float):
        """Clean up old rate limit entries."""
        cutoff_time = current_time - 3600  # Remove entries older than 1 hour
        
        keys_to_remove = []
        for key, request_times in self.requests.items():
            # Remove old requests
            request_times[:] = [t for t in request_times if t > cutoff_time]
            
            # Remove empty entries
            if not request_times:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.requests[key]
        
        self.last_cleanup = current_time


# Global rate limiter instance
rate_limiter = RateLimiter()


# ===== AUTHENTICATION DEPENDENCIES =====

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_bearer)
) -> Optional[Dict[str, Any]]:
    """
    Get current authenticated user from JWT token.
    
    Args:
        credentials: HTTP Bearer credentials
        
    Returns:
        Optional[Dict]: User information or None
        
    Raises:
        HTTPException: If authentication fails
    """
    if not credentials:
        return None
    
    try:
        payload = verify_token(credentials.credentials, "access")
        user_id = payload.get("sub")
        
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: missing user ID",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        # Here you would typically fetch user from database
        # For now, return the payload
        return {
            "id": user_id,
            "email": payload.get("email"),
            "username": payload.get("username"),
            "is_active": payload.get("is_active", True),
            "roles": payload.get("roles", [])
        }
        
    except (TokenExpiredError, CustomInvalidTokenError, AuthenticationError) as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"}
        )


async def get_current_active_user(
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get current active user.
    
    Args:
        current_user: Current user from get_current_user
        
    Returns:
        Dict: Active user information
        
    Raises:
        HTTPException: If user is not active or not authenticated
    """
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    if not current_user.get("is_active", False):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    
    return current_user


async def require_api_key(api_key: Optional[str] = Depends(api_key_header)) -> str:
    """
    Require valid API key.
    
    Args:
        api_key: API key from header
        
    Returns:
        str: Valid API key
        
    Raises:
        HTTPException: If API key is invalid
    """
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required"
        )
    
    # Here you would verify the API key against stored hashes
    # For now, just check if it has the correct format
    if not api_key.startswith("rsc_"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key format"
        )
    
    return api_key


async def require_admin_role(
    current_user: Dict[str, Any] = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Require admin role.
    
    Args:
        current_user: Current active user
        
    Returns:
        Dict: User with admin role
        
    Raises:
        HTTPException: If user doesn't have admin role
    """
    user_roles = current_user.get("roles", [])
    
    if "admin" not in user_roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role required"
        )
    
    return current_user


# ===== SECURITY MANAGER =====

class SecurityManager:
    """Central security management class."""
    
    def __init__(self):
        """Initialize security manager."""
        self.token_manager = TokenManager()
        self.rate_limiter = RateLimiter()
        self.failed_login_attempts = {}
        self.max_login_attempts = 5
        self.lockout_duration = 900  # 15 minutes
    
    def authenticate_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """
        Authenticate user with username and password.
        
        Args:
            username: Username
            password: Password
            
        Returns:
            Optional[Dict]: User information if authenticated
        """
        # Check for account lockout
        if self.is_account_locked(username):
            raise AuthenticationError("Account temporarily locked due to failed login attempts")
        
        # Here you would fetch user from database and verify password
        # For now, return None (not implemented)
        
        # If authentication fails, record failed attempt
        self.record_failed_login(username)
        return None
    
    def is_account_locked(self, username: str) -> bool:
        """
        Check if account is locked due to failed login attempts.
        
        Args:
            username: Username to check
            
        Returns:
            bool: True if account is locked
        """
        if username not in self.failed_login_attempts:
            return False
        
        attempts_data = self.failed_login_attempts[username]
        
        # Check if lockout period has expired
        if time.time() - attempts_data["last_attempt"] > self.lockout_duration:
            del self.failed_login_attempts[username]
            return False
        
        return attempts_data["count"] >= self.max_login_attempts
    
    def record_failed_login(self, username: str):
        """
        Record failed login attempt.
        
        Args:
            username: Username that failed login
        """
        current_time = time.time()
        
        if username in self.failed_login_attempts:
            attempts_data = self.failed_login_attempts[username]
            
            # Reset counter if enough time has passed
            if current_time - attempts_data["last_attempt"] > self.lockout_duration:
                attempts_data["count"] = 1
            else:
                attempts_data["count"] += 1
            
            attempts_data["last_attempt"] = current_time
        else:
            self.failed_login_attempts[username] = {
                "count": 1,
                "last_attempt": current_time
            }
    
    def clear_failed_login_attempts(self, username: str):
        """
        Clear failed login attempts for user.
        
        Args:
            username: Username to clear attempts for
        """
        if username in self.failed_login_attempts:
            del self.failed_login_attempts[username]
    
    def check_rate_limit(self, identifier: str, endpoint: str, limit: int = 60, window: int = 60) -> bool:
        """
        Check rate limit for identifier and endpoint.
        
        Args:
            identifier: User/IP identifier
            endpoint: API endpoint
            limit: Request limit
            window: Time window in seconds
            
        Returns:
            bool: True if request is allowed
        """
        key = rate_limit_key(identifier, endpoint)
        return self.rate_limiter.is_allowed(key, limit, window)


# Global security manager instance
security_manager = SecurityManager()


# ===== DECORATORS =====

def require_auth(func):
    """Decorator to require authentication."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # This would be implemented based on your specific needs
        return await func(*args, **kwargs)
    return wrapper


def require_role(role: str):
    """Decorator to require specific role."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # This would be implemented based on your specific needs
            return await func(*args, **kwargs)
        return wrapper
    return decorator


def rate_limit(limit: int = 60, window: int = 60):
    """Decorator for rate limiting."""
    def decorator(func):
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            client_ip = request.client.host if request.client else "unknown"
            endpoint = request.url.path
            
            if not security_manager.check_rate_limit(client_ip, endpoint, limit, window):
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded"
                )
            
            return await func(request, *args, **kwargs)
        return wrapper
    return decorator
