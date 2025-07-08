"""
Logging Configuration for AI Resume Screener

This module sets up comprehensive logging configuration including
file rotation, formatting, and different log levels for various components.
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from typing import Optional
from pathlib import Path

from app.config.settings import settings


class ColoredFormatter(logging.Formatter):
    """
    Custom formatter that adds colors to log levels for console output.
    """
    
    # Color codes for different log levels
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset color
    }
    
    def format(self, record):
        """Format log record with colors"""
        if record.levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[record.levelname]}{record.levelname}"
                f"{self.COLORS['RESET']}"
            )
        return super().format(record)


class RequestFormatter(logging.Formatter):
    """
    Custom formatter for HTTP request logging.
    """
    
    def format(self, record):
        """Format HTTP request log records"""
        if hasattr(record, 'method') and hasattr(record, 'url'):
            record.msg = f"{record.method} {record.url} - {record.msg}"
        return super().format(record)


def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
    enable_console: bool = True,
    enable_file: bool = True
) -> None:
    """
    Setup comprehensive logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file
        enable_console: Enable console logging
        enable_file: Enable file logging
    """
    
    # Use settings defaults if not provided
    log_level = log_level or settings.LOG_LEVEL
    log_file = log_file or settings.LOG_FILE
    
    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Define log format
    detailed_format = (
        "%(asctime)s | %(name)s | %(levelname)s | "
        "%(filename)s:%(lineno)d | %(funcName)s | %(message)s"
    )
    
    simple_format = (
        "%(asctime)s | %(levelname)s | %(message)s"
    )
    
    # Console handler with colors
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        
        if settings.is_development:
            console_formatter = ColoredFormatter(
                fmt=detailed_format,
                datefmt="%Y-%m-%d %H:%M:%S"
            )
        else:
            console_formatter = logging.Formatter(
                fmt=simple_format,
                datefmt="%Y-%m-%d %H:%M:%S"
            )
        
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
    
    # File handler with rotation
    if enable_file:
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_file,
            maxBytes=settings.LOG_MAX_SIZE,
            backupCount=settings.LOG_BACKUP_COUNT,
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, log_level.upper()))
        
        file_formatter = logging.Formatter(
            fmt=detailed_format,
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    # Configure specific loggers
    configure_specific_loggers()
    
    # Log startup message
    logger = logging.getLogger(__name__)
    logger.info(f"🚀 Logging configured - Level: {log_level}, File: {log_file}")


def configure_specific_loggers():
    """Configure specific loggers for different components"""
    
    # FastAPI/Uvicorn loggers
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    
    # Reduce noise from external libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    # spaCy logger
    logging.getLogger("spacy").setLevel(logging.WARNING)
    
    # Scikit-learn logger
    logging.getLogger("sklearn").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        logging.Logger: Configured logger instance
    """
    return logging.getLogger(name)


def log_request(
    method: str,
    url: str,
    status_code: int,
    process_time: float,
    client_ip: str = "unknown"
) -> None:
    """
    Log HTTP request information.
    
    Args:
        method: HTTP method
        url: Request URL
        status_code: HTTP status code
        process_time: Request processing time in seconds
        client_ip: Client IP address
    """
    logger = get_logger("request")
    
    # Determine log level based on status code
    if status_code >= 500:
        log_level = logging.ERROR
    elif status_code >= 400:
        log_level = logging.WARNING
    else:
        log_level = logging.INFO
    
    message = (
        f"{method} {url} - {status_code} - "
        f"{process_time:.3f}s - {client_ip}"
    )
    
    logger.log(log_level, message)


def log_error(
    error: Exception,
    context: str = "",
    extra_data: Optional[dict] = None
) -> None:
    """
    Log error with context and additional data.
    
    Args:
        error: Exception instance
        context: Context where error occurred
        extra_data: Additional data to log
    """
    logger = get_logger("error")
    
    error_msg = f"Error in {context}: {str(error)}"
    
    if extra_data:
        error_msg += f" | Extra data: {extra_data}"
    
    logger.error(error_msg, exc_info=True)


def log_performance(
    operation: str,
    duration: float,
    details: Optional[dict] = None
) -> None:
    """
    Log performance metrics.
    
    Args:
        operation: Operation name
        duration: Operation duration in seconds
        details: Additional performance details
    """
    logger = get_logger("performance")
    
    message = f"Performance: {operation} took {duration:.3f}s"
    
    if details:
        message += f" | Details: {details}"
    
    logger.info(message)


# Custom log levels
TRACE_LEVEL = 5
logging.addLevelName(TRACE_LEVEL, "TRACE")

def trace(self, message, *args, **kwargs):
    """Add TRACE level logging method"""
    if self.isEnabledFor(TRACE_LEVEL):
        self._log(TRACE_LEVEL, message, args, **kwargs)

logging.Logger.trace = trace
