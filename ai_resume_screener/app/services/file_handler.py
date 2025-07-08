"""
File Handler Service for AI Resume Screener

This module handles file upload, download, validation, and management
operations for resume files and other documents.
"""

import os
import shutil
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path
import asyncio
import aiofiles
from datetime import datetime, timedelta
import hashlib
import mimetypes

from fastapi import UploadFile
from app.config.settings import settings
from app.utils.exceptions import FileProcessingError

# Setup logging
logger = logging.getLogger(__name__)


class FileHandler:
    """
    File handling service for upload, download, and management operations.
    """
    
    def __init__(self):
        """Initialize file handler with upload directory."""
        self.upload_dir = Path(settings.UPLOAD_DIR)
        self.upload_dir.mkdir(exist_ok=True)
        logger.info(f"File handler initialized with upload directory: {self.upload_dir}")
    
    async def save_file(self, file: UploadFile, file_path: str) -> Dict[str, Any]:
        """
        Save uploaded file to specified path.
        
        Args:
            file: FastAPI UploadFile object
            file_path: Path where file should be saved
            
        Returns:
            Dict containing file information
            
        Raises:
            FileProcessingError: If file saving fails
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Save file
            async with aiofiles.open(file_path, 'wb') as f:
                content = await file.read()
                await f.write(content)
            
            # Get file info
            file_info = await self.get_file_info(file_path)
            file_info.update({
                "original_filename": file.filename,
                "content_type": file.content_type,
                "upload_timestamp": datetime.utcnow().isoformat()
            })
            
            logger.info(f"File saved successfully: {file_path}")
            return file_info
            
        except Exception as e:
            logger.error(f"Failed to save file: {str(e)}")
            raise FileProcessingError(f"Failed to save file: {str(e)}")
    
    async def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get information about a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dict containing file information
        """
        try:
            if not os.path.exists(file_path):
                raise FileProcessingError(f"File not found: {file_path}")
            
            file_stats = os.stat(file_path)
            file_path_obj = Path(file_path)
            
            # Get MIME type
            mime_type, _ = mimetypes.guess_type(file_path)
            
            # Calculate file hash
            file_hash = await self._calculate_file_hash(file_path)
            
            return {
                "file_path": file_path,
                "filename": file_path_obj.name,
                "file_size": file_stats.st_size,
                "file_size_mb": round(file_stats.st_size / (1024 * 1024), 2),
                "extension": file_path_obj.suffix.lower(),
                "mime_type": mime_type,
                "created_at": datetime.fromtimestamp(file_stats.st_ctime).isoformat(),
                "modified_at": datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                "file_hash": file_hash
            }
            
        except Exception as e:
            logger.error(f"Failed to get file info: {str(e)}")
            return {}
    
    async def _calculate_file_hash(self, file_path: str) -> str:
        """
        Calculate SHA-256 hash of file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            str: File hash
        """
        try:
            hash_sha256 = hashlib.sha256()
            async with aiofiles.open(file_path, 'rb') as f:
                while chunk := await f.read(8192):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception:
            return ""
    
    async def delete_file(self, file_path: str) -> bool:
        """
        Delete a file.
        
        Args:
            file_path: Path to the file to delete
            
        Returns:
            bool: True if file was deleted successfully
        """
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"File deleted: {file_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete file {file_path}: {str(e)}")
            return False
    
    async def move_file(self, source_path: str, destination_path: str) -> bool:
        """
        Move file from source to destination.
        
        Args:
            source_path: Source file path
            destination_path: Destination file path
            
        Returns:
            bool: True if file was moved successfully
        """
        try:
            # Ensure destination directory exists
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
            
            shutil.move(source_path, destination_path)
            logger.info(f"File moved from {source_path} to {destination_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to move file: {str(e)}")
            return False
    
    async def copy_file(self, source_path: str, destination_path: str) -> bool:
        """
        Copy file from source to destination.
        
        Args:
            source_path: Source file path
            destination_path: Destination file path
            
        Returns:
            bool: True if file was copied successfully
        """
        try:
            # Ensure destination directory exists
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
            
            shutil.copy2(source_path, destination_path)
            logger.info(f"File copied from {source_path} to {destination_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to copy file: {str(e)}")
            return False
    
    async def list_files(self, directory: str = None) -> List[Dict[str, Any]]:
        """
        List files in directory.
        
        Args:
            directory: Directory to list files from (defaults to upload directory)
            
        Returns:
            List of file information dictionaries
        """
        if directory is None:
            directory = str(self.upload_dir)
        
        try:
            files = []
            if os.path.exists(directory):
                for filename in os.listdir(directory):
                    file_path = os.path.join(directory, filename)
                    if os.path.isfile(file_path):
                        file_info = await self.get_file_info(file_path)
                        files.append(file_info)
            
            # Sort by modification time (newest first)
            files.sort(key=lambda x: x.get("modified_at", ""), reverse=True)
            return files
            
        except Exception as e:
            logger.error(f"Failed to list files: {str(e)}")
            return []
    
    async def cleanup_old_files(self, max_age_hours: int = 24) -> Dict[str, Any]:
        """
        Clean up old files from upload directory.
        
        Args:
            max_age_hours: Maximum age of files in hours
            
        Returns:
            Dict containing cleanup results
        """
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
            deleted_files = []
            total_size_freed = 0
            
            files = await self.list_files()
            
            for file_info in files:
                file_path = file_info.get("file_path")
                modified_at = datetime.fromisoformat(file_info.get("modified_at", ""))
                
                if modified_at < cutoff_time:
                    if await self.delete_file(file_path):
                        deleted_files.append(file_info["filename"])
                        total_size_freed += file_info.get("file_size", 0)
            
            logger.info(f"Cleanup completed: {len(deleted_files)} files deleted, "
                       f"{total_size_freed / (1024*1024):.2f} MB freed")
            
            return {
                "deleted_files": deleted_files,
                "files_deleted": len(deleted_files),
                "total_size_freed_mb": round(total_size_freed / (1024*1024), 2),
                "cleanup_timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")
            return {
                "error": str(e),
                "files_deleted": 0,
                "total_size_freed_mb": 0
            }
    
    async def get_directory_size(self, directory: str = None) -> Dict[str, Any]:
        """
        Get size information for directory.
        
        Args:
            directory: Directory to analyze (defaults to upload directory)
            
        Returns:
            Dict containing size information
        """
        if directory is None:
            directory = str(self.upload_dir)
        
        try:
            total_size = 0
            file_count = 0
            
            if os.path.exists(directory):
                for dirpath, dirnames, filenames in os.walk(directory):
                    for filename in filenames:
                        file_path = os.path.join(dirpath, filename)
                        try:
                            file_size = os.path.getsize(file_path)
                            total_size += file_size
                            file_count += 1
                        except OSError:
                            continue
            
            return {
                "directory": directory,
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024*1024), 2),
                "file_count": file_count,
                "average_file_size_mb": round((total_size / file_count) / (1024*1024), 2) if file_count > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to get directory size: {str(e)}")
            return {
                "error": str(e),
                "total_size_bytes": 0,
                "total_size_mb": 0,
                "file_count": 0
            }
    
    async def validate_file_integrity(self, file_path: str, expected_hash: str = None) -> bool:
        """
        Validate file integrity.
        
        Args:
            file_path: Path to the file
            expected_hash: Expected file hash (optional)
            
        Returns:
            bool: True if file is valid
        """
        try:
            if not os.path.exists(file_path):
                return False
            
            # Check if file is readable
            try:
                async with aiofiles.open(file_path, 'rb') as f:
                    # Try to read the entire file to check readability
                    while await f.read(8192):
                        pass
            except Exception:
                return False
            
            # If expected hash is provided, verify it
            if expected_hash:
                actual_hash = await self._calculate_file_hash(file_path)
                if actual_hash != expected_hash:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"File integrity validation failed: {str(e)}")
            return False
    
    async def create_backup(self, file_path: str, backup_dir: str = None) -> Optional[str]:
        """
        Create a backup of the file.
        
        Args:
            file_path: Path to the file to backup
            backup_dir: Directory to store backup (optional)
            
        Returns:
            Optional[str]: Path to backup file if successful
        """
        try:
            if not os.path.exists(file_path):
                return None
            
            if backup_dir is None:
                backup_dir = os.path.join(str(self.upload_dir), "backups")
            
            os.makedirs(backup_dir, exist_ok=True)
            
            # Create backup filename with timestamp
            file_name = Path(file_path).name
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"{timestamp}_{file_name}"
            backup_path = os.path.join(backup_dir, backup_filename)
            
            # Copy file to backup location
            if await self.copy_file(file_path, backup_path):
                logger.info(f"Backup created: {backup_path}")
                return backup_path
            
            return None
            
        except Exception as e:
            logger.error(f"Backup creation failed: {str(e)}")
            return None
    
    async def restore_from_backup(self, backup_path: str, restore_path: str) -> bool:
        """
        Restore file from backup.
        
        Args:
            backup_path: Path to backup file
            restore_path: Path where file should be restored
            
        Returns:
            bool: True if restore was successful
        """
        try:
            if not os.path.exists(backup_path):
                return False
            
            success = await self.copy_file(backup_path, restore_path)
            if success:
                logger.info(f"File restored from backup: {backup_path} -> {restore_path}")
            
            return success
            
        except Exception as e:
            logger.error(f"Restore from backup failed: {str(e)}")
            return False
    
    async def get_file_permissions(self, file_path: str) -> Dict[str, bool]:
        """
        Get file permissions.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dict containing permission information
        """
        try:
            if not os.path.exists(file_path):
                return {}
            
            return {
                "readable": os.access(file_path, os.R_OK),
                "writable": os.access(file_path, os.W_OK),
                "executable": os.access(file_path, os.X_OK),
                "exists": True
            }
            
        except Exception as e:
            logger.error(f"Failed to get file permissions: {str(e)}")
            return {"exists": False}
    
    async def health_check(self) -> bool:
        """
        Check if file handler service is healthy.
        
        Returns:
            bool: True if service is healthy
        """
        try:
            # Check if upload directory exists and is writable
            if not self.upload_dir.exists():
                return False
            
            if not os.access(str(self.upload_dir), os.W_OK):
                return False
            
            # Try to create a test file
            test_file = self.upload_dir / "health_check.tmp"
            try:
                async with aiofiles.open(str(test_file), 'w') as f:
                    await f.write("health check")
                
                # Clean up test file
                if test_file.exists():
                    test_file.unlink()
                
                return True
                
            except Exception:
                return False
                
        except Exception:
            return False
