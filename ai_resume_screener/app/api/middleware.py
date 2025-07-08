"""
Custom Middleware for AI Resume Screener

This module contains custom middleware for logging, error handling,
performance monitoring, and request processing.
"""

import time
import uuid
import logging
from typing import Callable, Dict, Any
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from app.config.settings import settings
from app.utils.exceptions import ResumeScreenerException

# Setup logging
logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for logging HTTP requests and responses.
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.logger = logging.getLogger("request")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request and log details.
        
        Args:
            request: HTTP request
            call_next: Next middleware/endpoint
            
        Returns:
            Response: HTTP response
        """
        # Generate request ID
        request_id = str(uuid.uuid4())
        
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        
        # Log request
        self.logger.info(
            f"Request {request_id}: {request.method} {request.url} from {client_ip}"
        )
        
        # Add request ID to request state
        request.state.request_id = request_id
        
        # Process request
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Log response
        self.logger.info(
            f"Response {request_id}: {response.status_code} "
            f"({process_time:.3f}s)"
        )
        
        return response


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for centralized error handling and response formatting.
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.logger = logging.getLogger("error")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Handle errors and format error responses.
        
        Args:
            request: HTTP request
            call_next: Next middleware/endpoint
            
        Returns:
            Response: HTTP response or error response
        """
        try:
            return await call_next(request)
            
        except ResumeScreenerException as e:
            # Handle custom application exceptions
            self.logger.error(
                f"Application error: {e.detail} (Code: {e.error_code})"
            )
            return JSONResponse(
                status_code=e.status_code,
                content={
                    "error": True,
                    "message": e.detail,
                    "error_code": e.error_code,
                    "timestamp": e.timestamp.isoformat(),
                    "request_id": getattr(request.state, "request_id", None)
                }
            )
            
        except Exception as e:
            # Handle unexpected exceptions
            self.logger.error(
                f"Unexpected error: {str(e)}",
                exc_info=True
            )
            return JSONResponse(
                status_code=500,
                content={
                    "error": True,
                    "message": "Internal server error",
                    "detail": str(e) if settings.is_development else None,
                    "request_id": getattr(request.state, "request_id", None)
                }
            )


class ProcessTimeMiddleware(BaseHTTPMiddleware):
    """
    Middleware for adding process time headers to responses.
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Add process time to response headers.
        
        Args:
            request: HTTP request
            call_next: Next middleware/endpoint
            
        Returns:
            Response: HTTP response with process time header
        """
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Add process time header
        response.headers["X-Process-Time"] = str(process_time)
        
        # Log slow requests
        if process_time > 5.0:  # Log requests taking more than 5 seconds
            logger.warning(
                f"Slow request: {request.method} {request.url} "
                f"took {process_time:.3f}s"
            )
        
        return response


class SecurityMiddleware(BaseHTTPMiddleware):
    """
    Middleware for adding security headers and basic security checks.
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Content-Security-Policy": "default-src 'self'"
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Add security headers to responses.
        
        Args:
            request: HTTP request
            call_next: Next middleware/endpoint
            
        Returns:
            Response: HTTP response with security headers
        """
        response = await call_next(request)
        
        # Add security headers
        for header, value in self.security_headers.items():
            response.headers[header] = value
        
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Basic rate limiting middleware (in-memory implementation).
    """
    
    def __init__(self, app: ASGIApp, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests: Dict[str, list] = {}
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Apply rate limiting based on client IP.
        
        Args:
            request: HTTP request
            call_next: Next middleware/endpoint
            
        Returns:
            Response: HTTP response or rate limit error
        """
        client_ip = request.client.host if request.client else "unknown"
        current_time = time.time()
        
        # Clean old requests
        if client_ip in self.requests:
            self.requests[client_ip] = [
                req_time for req_time in self.requests[client_ip]
                if current_time - req_time < 60  # Keep requests from last minute
            ]
        else:
            self.requests[client_ip] = []
        
        # Check rate limit
        if len(self.requests[client_ip]) >= self.requests_per_minute:
            return JSONResponse(
                status_code=429,
                content={
                    "error": True,
                    "message": "Rate limit exceeded",
                    "detail": f"Maximum {self.requests_per_minute} requests per minute"
                }
            )
        
        # Add current request
        self.requests[client_ip].append(current_time)
        
        return await call_next(request)


class CacheMiddleware(BaseHTTPMiddleware):
    """
    Simple caching middleware for GET requests.
    """
    
    def __init__(self, app: ASGIApp, cache_ttl: int = 300):
        super().__init__(app)
        self.cache_ttl = cache_ttl
        self.cache: Dict[str, Dict[str, Any]] = {}
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Cache GET requests for specified TTL.
        
        Args:
            request: HTTP request
            call_next: Next middleware/endpoint
            
        Returns:
            Response: Cached response or fresh response
        """
        # Only cache GET requests
        if request.method != "GET":
            return await call_next(request)
        
        # Generate cache key
        cache_key = f"{request.method}:{request.url}"
        current_time = time.time()
        
        # Check cache
        if cache_key in self.cache:
            cached_data = self.cache[cache_key]
            if current_time - cached_data["timestamp"] < self.cache_ttl:
                # Return cached response
                return JSONResponse(
                    status_code=cached_data["status_code"],
                    content=cached_data["content"],
                    headers={"X-Cache": "HIT"}
                )
        
        # Get fresh response
        response = await call_next(request)
        
        # Cache successful responses
        if response.status_code == 200:
            # Read response content
            response_body = b""
            async for chunk in response.body_iterator:
                response_body += chunk
            
            # Store in cache
            self.cache[cache_key] = {
                "status_code": response.status_code,
                "content": response_body.decode(),
                "timestamp": current_time
            }
            
            # Create new response
            response = JSONResponse(
                status_code=response.status_code,
                content=response_body.decode(),
                headers={"X-Cache": "MISS"}
            )
        
        return response


class MetricsMiddleware(BaseHTTPMiddleware):
    """
    Middleware for collecting basic metrics.
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.metrics = {
            "total_requests": 0,
            "total_errors": 0,
            "average_response_time": 0.0,
            "endpoint_stats": {}
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Collect request metrics.
        
        Args:
            request: HTTP request
            call_next: Next middleware/endpoint
            
        Returns:
            Response: HTTP response with metrics collection
        """
        start_time = time.time()
        
        # Process request
        response = await call_next(request)
        
        # Calculate metrics
        process_time = time.time() - start_time
        endpoint = f"{request.method} {request.url.path}"
        
        # Update metrics
        self.metrics["total_requests"] += 1
        
        if response.status_code >= 400:
            self.metrics["total_errors"] += 1
        
        # Update average response time
        current_avg = self.metrics["average_response_time"]
        total_requests = self.metrics["total_requests"]
        self.metrics["average_response_time"] = (
            (current_avg * (total_requests - 1) + process_time) / total_requests
        )
        
        # Update endpoint stats
        if endpoint not in self.metrics["endpoint_stats"]:
            self.metrics["endpoint_stats"][endpoint] = {
                "count": 0,
                "avg_time": 0.0,
                "errors": 0
            }
        
        endpoint_stats = self.metrics["endpoint_stats"][endpoint]
        endpoint_stats["count"] += 1
        endpoint_stats["avg_time"] = (
            (endpoint_stats["avg_time"] * (endpoint_stats["count"] - 1) + process_time)
            / endpoint_stats["count"]
        )
        
        if response.status_code >= 400:
            endpoint_stats["errors"] += 1
        
        return response
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        return self.metrics.copy()
