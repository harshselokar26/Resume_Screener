"""
Helper Utility Functions for AI Resume Screener

This module contains various utility functions used throughout the application
for common operations like file handling, text processing, and data formatting.
"""

import os
import re
import uuid
import hashlib
import logging
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Union
from pathlib import Path
import json
import unicodedata

from app.config.settings import settings

# Setup logging
logger = logging.getLogger(__name__)


# ===== FILE UTILITIES =====

def generate_unique_filename(original_filename: str) -> str:
    """
    Generate a unique filename with timestamp and UUID.
    
    Args:
        original_filename: Original filename
        
    Returns:
        str: Unique filename
    """
    if not original_filename:
        original_filename = "unknown_file"
    
    # Get file extension
    file_extension = get_file_extension(original_filename)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generate short UUID
    short_uuid = str(uuid.uuid4())[:8]
    
    # Sanitize base filename
    base_name = sanitize_filename(Path(original_filename).stem)
    
    # Combine all parts
    unique_filename = f"{timestamp}_{short_uuid}_{base_name}{file_extension}"
    
    return unique_filename


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename by removing invalid characters.
    
    Args:
        filename: Original filename
        
    Returns:
        str: Sanitized filename
    """
    if not filename:
        return "unnamed_file"
    
    # Remove or replace invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove control characters
    filename = ''.join(char for char in filename if ord(char) >= 32)
    
    # Normalize unicode characters
    filename = unicodedata.normalize('NFKD', filename)
    
    # Remove extra spaces and dots
    filename = re.sub(r'\s+', '_', filename.strip())
    filename = re.sub(r'\.+', '.', filename)
    
    # Ensure filename is not too long
    if len(filename) > 100:
        filename = filename[:100]
    
    return filename or "unnamed_file"


def get_file_extension(filename: str) -> str:
    """
    Get file extension from filename.
    
    Args:
        filename: Filename
        
    Returns:
        str: File extension with dot (e.g., '.pdf')
    """
    if not filename:
        return ""
    
    return Path(filename).suffix.lower()


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: File size in bytes
        
    Returns:
        str: Formatted file size
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    size = float(size_bytes)
    
    while size >= 1024.0 and i < len(size_names) - 1:
        size /= 1024.0
        i += 1
    
    return f"{size:.1f} {size_names[i]}"


def cleanup_temp_files(file_paths: List[str]) -> int:
    """
    Clean up temporary files.
    
    Args:
        file_paths: List of file paths to delete
        
    Returns:
        int: Number of files successfully deleted
    """
    deleted_count = 0
    
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                deleted_count += 1
                logger.info(f"Deleted temporary file: {file_path}")
        except Exception as e:
            logger.error(f"Failed to delete file {file_path}: {str(e)}")
    
    return deleted_count


# ===== TEXT PROCESSING UTILITIES =====

def extract_email_from_text(text: str) -> List[str]:
    """
    Extract email addresses from text.
    
    Args:
        text: Text to search for emails
        
    Returns:
        List[str]: List of found email addresses
    """
    if not text:
        return []
    
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, text)
    
    # Remove duplicates and validate
    unique_emails = []
    for email in emails:
        if email.lower() not in [e.lower() for e in unique_emails]:
            unique_emails.append(email)
    
    return unique_emails


def extract_phone_from_text(text: str) -> List[str]:
    """
    Extract phone numbers from text.
    
    Args:
        text: Text to search for phone numbers
        
    Returns:
        List[str]: List of found phone numbers
    """
    if not text:
        return []
    
    # Various phone number patterns
    phone_patterns = [
        r'\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',  # US format
        r'\+?\d{1,3}[-.\s]?\d{3,4}[-.\s]?\d{3,4}[-.\s]?\d{3,4}',  # International
        r'\(\d{3}\)\s?\d{3}-?\d{4}',  # (123) 456-7890
        r'\d{3}-\d{3}-\d{4}',  # 123-456-7890
        r'\d{10}',  # 1234567890
    ]
    
    phones = []
    for pattern in phone_patterns:
        matches = re.findall(pattern, text)
        phones.extend(matches)
    
    # Remove duplicates
    unique_phones = list(set(phones))
    
    return unique_phones


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to specified length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add when truncated
        
    Returns:
        str: Truncated text
    """
    if not text:
        return ""
    
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def normalize_skill_name(skill: str) -> str:
    """
    Normalize skill name for consistent comparison.
    
    Args:
        skill: Skill name
        
    Returns:
        str: Normalized skill name
    """
    if not skill:
        return ""
    
    # Convert to lowercase and strip whitespace
    normalized = skill.lower().strip()
    
    # Remove extra spaces
    normalized = re.sub(r'\s+', ' ', normalized)
    
    # Handle common variations
    skill_mappings = {
        'javascript': 'javascript',
        'js': 'javascript',
        'nodejs': 'node.js',
        'node': 'node.js',
        'reactjs': 'react',
        'react.js': 'react',
        'vuejs': 'vue.js',
        'vue': 'vue.js',
        'python3': 'python',
        'py': 'python',
        'c++': 'cpp',
        'c#': 'csharp',
        'c sharp': 'csharp',
        'postgresql': 'postgres',
        'mysql': 'mysql',
        'mongodb': 'mongo',
        'aws': 'amazon web services',
        'gcp': 'google cloud platform',
        'azure': 'microsoft azure'
    }
    
    return skill_mappings.get(normalized, normalized)


def calculate_reading_time(text: str, words_per_minute: int = 200) -> int:
    """
    Calculate estimated reading time for text.
    
    Args:
        text: Text to analyze
        words_per_minute: Average reading speed
        
    Returns:
        int: Estimated reading time in minutes
    """
    if not text:
        return 0
    
    word_count = len(text.split())
    reading_time = max(1, round(word_count / words_per_minute))
    
    return reading_time


def hash_text(text: str, algorithm: str = "sha256") -> str:
    """
    Generate hash of text.
    
    Args:
        text: Text to hash
        algorithm: Hash algorithm (md5, sha1, sha256)
        
    Returns:
        str: Hash string
    """
    if not text:
        return ""
    
    text_bytes = text.encode('utf-8')
    
    if algorithm == "md5":
        return hashlib.md5(text_bytes).hexdigest()
    elif algorithm == "sha1":
        return hashlib.sha1(text_bytes).hexdigest()
    elif algorithm == "sha256":
        return hashlib.sha256(text_bytes).hexdigest()
    else:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")


def mask_sensitive_data(text: str, mask_char: str = "*") -> str:
    """
    Mask sensitive data in text (emails, phones, etc.).
    
    Args:
        text: Text containing sensitive data
        mask_char: Character to use for masking
        
    Returns:
        str: Text with masked sensitive data
    """
    if not text:
        return ""
    
    masked_text = text
    
    # Mask email addresses
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    masked_text = re.sub(email_pattern, lambda m: m.group()[:2] + mask_char * (len(m.group()) - 4) + m.group()[-2:], masked_text)
    
    # Mask phone numbers
    phone_pattern = r'\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
    masked_text = re.sub(phone_pattern, lambda m: mask_char * len(m.group()), masked_text)
    
    return masked_text


# ===== DATA PROCESSING UTILITIES =====

def calculate_similarity_percentage(score: float) -> float:
    """
    Convert similarity score to percentage.
    
    Args:
        score: Similarity score (0-1)
        
    Returns:
        float: Percentage (0-100)
    """
    if not isinstance(score, (int, float)):
        return 0.0
    
    percentage = max(0.0, min(100.0, score * 100))
    return round(percentage, 2)


def is_valid_uuid(uuid_string: str) -> bool:
    """
    Check if string is a valid UUID.
    
    Args:
        uuid_string: String to validate
        
    Returns:
        bool: True if valid UUID
    """
    if not uuid_string:
        return False
    
    try:
        uuid.UUID(uuid_string)
        return True
    except ValueError:
        return False


def get_current_timestamp() -> str:
    """
    Get current timestamp in ISO format.
    
    Returns:
        str: Current timestamp
    """
    return datetime.now(timezone.utc).isoformat()


def safe_json_loads(json_string: str, default: Any = None) -> Any:
    """
    Safely parse JSON string.
    
    Args:
        json_string: JSON string to parse
        default: Default value if parsing fails
        
    Returns:
        Any: Parsed JSON or default value
    """
    if not json_string:
        return default
    
    try:
        return json.loads(json_string)
    except (json.JSONDecodeError, TypeError):
        return default


def safe_json_dumps(obj: Any, default: str = "{}") -> str:
    """
    Safely serialize object to JSON.
    
    Args:
        obj: Object to serialize
        default: Default value if serialization fails
        
    Returns:
        str: JSON string or default value
    """
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return default


def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """
    Flatten nested dictionary.
    
    Args:
        d: Dictionary to flatten
        parent_key: Parent key prefix
        sep: Separator for nested keys
        
    Returns:
        Dict[str, Any]: Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split list into chunks of specified size.
    
    Args:
        lst: List to chunk
        chunk_size: Size of each chunk
        
    Returns:
        List[List[Any]]: List of chunks
    """
    if chunk_size <= 0:
        return [lst]
    
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def remove_duplicates_preserve_order(lst: List[Any]) -> List[Any]:
    """
    Remove duplicates from list while preserving order.
    
    Args:
        lst: List with potential duplicates
        
    Returns:
        List[Any]: List without duplicates
    """
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """
    Calculate percentage change between two values.
    
    Args:
        old_value: Original value
        new_value: New value
        
    Returns:
        float: Percentage change
    """
    if old_value == 0:
        return 100.0 if new_value > 0 else 0.0
    
    change = ((new_value - old_value) / old_value) * 100
    return round(change, 2)


def retry_operation(func, max_retries: int = 3, delay: float = 1.0):
    """
    Retry operation with exponential backoff.
    
    Args:
        func: Function to retry
        max_retries: Maximum number of retries
        delay: Initial delay between retries
        
    Returns:
        Any: Function result
    """
    import time
    
    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries:
                raise e
            
            wait_time = delay * (2 ** attempt)
            logger.warning(f"Operation failed (attempt {attempt + 1}/{max_retries + 1}), retrying in {wait_time}s: {str(e)}")
            time.sleep(wait_time)
