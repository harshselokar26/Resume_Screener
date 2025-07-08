"""
Input Validation Functions for AI Resume Screener

This module contains validation functions for various types of input data
including files, text, emails, URLs, and other data formats.
"""

import re
import os
import json
import uuid
from datetime import datetime
from typing import Any, List, Dict, Optional, Union
from pathlib import Path
import mimetypes

from app.config.settings import settings


# ===== FILE VALIDATION =====

def validate_file_type(filename: str) -> bool:
    """
    Validate if file type is allowed.
    
    Args:
        filename: Name of the file
        
    Returns:
        bool: True if file type is allowed
    """
    if not filename:
        return False
    
    file_extension = Path(filename).suffix.lower().lstrip('.')
    return file_extension in [ext.lower() for ext in settings.ALLOWED_FILE_TYPES]


def validate_file_size(file_size: int) -> bool:
    """
    Validate if file size is within allowed limits.
    
    Args:
        file_size: Size of file in bytes
        
    Returns:
        bool: True if file size is allowed
    """
    if file_size <= 0:
        return False
    
    return file_size <= settings.MAX_FILE_SIZE


def validate_file_content(file_path: str) -> bool:
    """
    Validate file content and structure.
    
    Args:
        file_path: Path to the file
        
    Returns:
        bool: True if file content is valid
    """
    if not os.path.exists(file_path):
        return False
    
    try:
        # Check if file is readable
        with open(file_path, 'rb') as f:
            # Read first few bytes to check file signature
            header = f.read(1024)
            
        # Check file signatures for common formats
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension == '.pdf':
            return header.startswith(b'%PDF-')
        elif file_extension == '.docx':
            return header.startswith(b'PK\x03\x04')  # ZIP signature
        elif file_extension == '.doc':
            return (header.startswith(b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1') or  # OLE signature
                   header.startswith(b'PK\x03\x04'))  # Some DOC files use ZIP
        
        return True
        
    except Exception:
        return False


def validate_mime_type(file_path: str, expected_types: List[str] = None) -> bool:
    """
    Validate MIME type of file.
    
    Args:
        file_path: Path to the file
        expected_types: List of expected MIME types
        
    Returns:
        bool: True if MIME type is valid
    """
    if not os.path.exists(file_path):
        return False
    
    mime_type, _ = mimetypes.guess_type(file_path)
    
    if expected_types is None:
        expected_types = [
            'application/pdf',
            'application/msword',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        ]
    
    return mime_type in expected_types


# ===== TEXT VALIDATION =====

def validate_text_length(text: str, min_length: int = 1, max_length: int = 50000) -> bool:
    """
    Validate text length.
    
    Args:
        text: Text to validate
        min_length: Minimum allowed length
        max_length: Maximum allowed length
        
    Returns:
        bool: True if text length is valid
    """
    if not isinstance(text, str):
        return False
    
    text_length = len(text.strip())
    return min_length <= text_length <= max_length


def validate_text_content(text: str, forbidden_patterns: List[str] = None) -> bool:
    """
    Validate text content for forbidden patterns.
    
    Args:
        text: Text to validate
        forbidden_patterns: List of forbidden regex patterns
        
    Returns:
        bool: True if text content is valid
    """
    if not isinstance(text, str):
        return False
    
    if forbidden_patterns is None:
        forbidden_patterns = [
            r'<script.*?>.*?</script>',  # Script tags
            r'javascript:',  # JavaScript URLs
            r'on\w+\s*=',  # Event handlers
            r'<iframe.*?>',  # Iframe tags
        ]
    
    for pattern in forbidden_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return False
    
    return True


def validate_skill_name(skill: str) -> bool:
    """
    Validate skill name format.
    
    Args:
        skill: Skill name to validate
        
    Returns:
        bool: True if skill name is valid
    """
    if not isinstance(skill, str):
        return False
    
    skill = skill.strip()
    
    # Check length
    if not (1 <= len(skill) <= 100):
        return False
    
    # Check for valid characters (letters, numbers, spaces, dots, hyphens, plus)
    if not re.match(r'^[a-zA-Z0-9\s\.\-\+#]+$', skill):
        return False
    
    # Check that it's not just whitespace or special characters
    if not re.search(r'[a-zA-Z0-9]', skill):
        return False
    
    return True


# ===== EMAIL AND CONTACT VALIDATION =====

def validate_email(email: str) -> bool:
    """
    Validate email address format.
    
    Args:
        email: Email address to validate
        
    Returns:
        bool: True if email is valid
    """
    if not isinstance(email, str):
        return False
    
    email = email.strip().lower()
    
    # Basic email regex pattern
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    if not re.match(email_pattern, email):
        return False
    
    # Additional checks
    if len(email) > 254:  # RFC 5321 limit
        return False
    
    local_part, domain = email.split('@')
    
    # Local part checks
    if len(local_part) > 64:  # RFC 5321 limit
        return False
    
    # Domain checks
    if len(domain) > 253:
        return False
    
    return True


def validate_phone(phone: str) -> bool:
    """
    Validate phone number format.
    
    Args:
        phone: Phone number to validate
        
    Returns:
        bool: True if phone number is valid
    """
    if not isinstance(phone, str):
        return False
    
    # Remove all non-digit characters
    digits_only = re.sub(r'\D', '', phone)
    
    # Check length (7-15 digits for international numbers)
    if not (7 <= len(digits_only) <= 15):
        return False
    
    # Common phone number patterns
    phone_patterns = [
        r'^\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}$',  # US
        r'^\+?[1-9]\d{1,14}$',  # International E.164 format
        r'^\d{10}$',  # 10 digits
        r'^\d{7}$',   # 7 digits
    ]
    
    return any(re.match(pattern, phone.strip()) for pattern in phone_patterns)


def validate_url(url: str) -> bool:
    """
    Validate URL format.
    
    Args:
        url: URL to validate
        
    Returns:
        bool: True if URL is valid
    """
    if not isinstance(url, str):
        return False
    
    url = url.strip()
    
    # Basic URL pattern
    url_pattern = r'^https?://(?:[-\w.])+(?:\:[0-9]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:\#(?:[\w.])*)?)?$'
    
    return re.match(url_pattern, url) is not None


# ===== NUMERIC VALIDATION =====

def validate_similarity_score(score: Union[int, float]) -> bool:
    """
    Validate similarity score is between 0 and 1.
    
    Args:
        score: Similarity score to validate
        
    Returns:
        bool: True if score is valid
    """
    if not isinstance(score, (int, float)):
        return False
    
    return 0.0 <= score <= 1.0


def validate_percentage(percentage: Union[int, float]) -> bool:
    """
    Validate percentage is between 0 and 100.
    
    Args:
        percentage: Percentage to validate
        
    Returns:
        bool: True if percentage is valid
    """
    if not isinstance(percentage, (int, float)):
        return False
    
    return 0.0 <= percentage <= 100.0


def validate_positive_integer(value: Union[int, str]) -> bool:
    """
    Validate positive integer value.
    
    Args:
        value: Value to validate
        
    Returns:
        bool: True if value is a positive integer
    """
    try:
        int_value = int(value)
        return int_value > 0
    except (ValueError, TypeError):
        return False


def validate_non_negative_number(value: Union[int, float, str]) -> bool:
    """
    Validate non-negative number.
    
    Args:
        value: Value to validate
        
    Returns:
        bool: True if value is non-negative
    """
    try:
        num_value = float(value)
        return num_value >= 0
    except (ValueError, TypeError):
        return False


# ===== DATE AND TIME VALIDATION =====

def validate_date_format(date_string: str, format_string: str = "%Y-%m-%d") -> bool:
    """
    Validate date string format.
    
    Args:
        date_string: Date string to validate
        format_string: Expected date format
        
    Returns:
        bool: True if date format is valid
    """
    if not isinstance(date_string, str):
        return False
    
    try:
        datetime.strptime(date_string, format_string)
        return True
    except ValueError:
        return False


def validate_iso_datetime(datetime_string: str) -> bool:
    """
    Validate ISO datetime format.
    
    Args:
        datetime_string: Datetime string to validate
        
    Returns:
        bool: True if datetime format is valid
    """
    if not isinstance(datetime_string, str):
        return False
    
    try:
        datetime.fromisoformat(datetime_string.replace('Z', '+00:00'))
        return True
    except ValueError:
        return False


# ===== UUID AND ID VALIDATION =====

def validate_uuid(uuid_string: str, version: int = None) -> bool:
    """
    Validate UUID format.
    
    Args:
        uuid_string: UUID string to validate
        version: Specific UUID version to validate (1-5)
        
    Returns:
        bool: True if UUID is valid
    """
    if not isinstance(uuid_string, str):
        return False
    
    try:
        uuid_obj = uuid.UUID(uuid_string)
        
        if version is not None:
            return uuid_obj.version == version
        
        return True
    except ValueError:
        return False


# ===== JSON VALIDATION =====

def validate_json_structure(json_string: str, required_keys: List[str] = None) -> bool:
    """
    Validate JSON structure and required keys.
    
    Args:
        json_string: JSON string to validate
        required_keys: List of required keys
        
    Returns:
        bool: True if JSON structure is valid
    """
    if not isinstance(json_string, str):
        return False
    
    try:
        data = json.loads(json_string)
        
        if not isinstance(data, dict):
            return False
        
        if required_keys:
            return all(key in data for key in required_keys)
        
        return True
    except (json.JSONDecodeError, TypeError):
        return False


def validate_json_schema(data: Dict[str, Any], schema: Dict[str, Any]) -> bool:
    """
    Validate data against a simple schema.
    
    Args:
        data: Data to validate
        schema: Schema definition
        
    Returns:
        bool: True if data matches schema
    """
    if not isinstance(data, dict) or not isinstance(schema, dict):
        return False
    
    try:
        for key, expected_type in schema.items():
            if key not in data:
                return False
            
            if not isinstance(data[key], expected_type):
                return False
        
        return True
    except Exception:
        return False


# ===== BATCH VALIDATION =====

def validate_batch_data(data_list: List[Any], validator_func, max_items: int = 100) -> Dict[str, Any]:
    """
    Validate batch of data items.
    
    Args:
        data_list: List of data items to validate
        validator_func: Validation function to apply
        max_items: Maximum number of items allowed
        
    Returns:
        Dict containing validation results
    """
    if not isinstance(data_list, list):
        return {
            "valid": False,
            "error": "Data must be a list",
            "valid_items": 0,
            "invalid_items": 0
        }
    
    if len(data_list) > max_items:
        return {
            "valid": False,
            "error": f"Too many items. Maximum allowed: {max_items}",
            "valid_items": 0,
            "invalid_items": len(data_list)
        }
    
    valid_items = 0
    invalid_items = 0
    errors = []
    
    for i, item in enumerate(data_list):
        try:
            if validator_func(item):
                valid_items += 1
            else:
                invalid_items += 1
                errors.append(f"Item {i}: Invalid format")
        except Exception as e:
            invalid_items += 1
            errors.append(f"Item {i}: {str(e)}")
    
    return {
        "valid": invalid_items == 0,
        "valid_items": valid_items,
        "invalid_items": invalid_items,
        "errors": errors[:10],  # Limit error messages
        "total_items": len(data_list)
    }


# ===== COMPOSITE VALIDATORS =====

def validate_resume_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate resume data structure.
    
    Args:
        data: Resume data to validate
        
    Returns:
        Dict containing validation results
    """
    errors = []
    
    # Required fields
    required_fields = ['extracted_text', 'original_filename']
    for field in required_fields:
        if field not in data:
            errors.append(f"Missing required field: {field}")
    
    # Validate text content
    if 'extracted_text' in data:
        if not validate_text_length(data['extracted_text'], min_length=10):
            errors.append("Extracted text is too short")
    
    # Validate filename
    if 'original_filename' in data:
        if not validate_file_type(data['original_filename']):
            errors.append("Invalid file type")
    
    # Validate file size if present
    if 'file_size' in data:
        if not validate_file_size(data['file_size']):
            errors.append("File size exceeds limit")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors
    }


def validate_scoring_request(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate scoring request data.
    
    Args:
        data: Scoring request data to validate
        
    Returns:
        Dict containing validation results
    """
    errors = []
    
    # Required fields
    required_fields = ['resume_text', 'job_description']
    for field in required_fields:
        if field not in data:
            errors.append(f"Missing required field: {field}")
        elif not validate_text_length(data[field], min_length=10):
            errors.append(f"{field} is too short")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors
    }
