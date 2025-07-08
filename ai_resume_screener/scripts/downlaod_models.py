#!/usr/bin/env python3
"""
AI Resume Screener - Model Download Script

This script downloads and installs required machine learning models
including spaCy language models and other dependencies.
"""

import os
import sys
import subprocess
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import urllib.request
import zipfile
import tarfile
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/model_download.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Model configurations
SPACY_MODELS = {
    "en_core_web_sm": {
        "version": "3.7.1",
        "description": "English pipeline optimized for CPU",
        "size": "~15MB",
        "required": True
    },
    "en_core_web_md": {
        "version": "3.7.1", 
        "description": "English pipeline with word vectors",
        "size": "~50MB",
        "required": False
    },
    "en_core_web_lg": {
        "version": "3.7.1",
        "description": "English pipeline with large word vectors",
        "size": "~750MB",
        "required": False
    }
}

NLTK_DATA = {
    "punkt": {
        "description": "Punkt tokenizer models",
        "required": True
    },
    "stopwords": {
        "description": "Stopwords corpus",
        "required": True
    },
    "wordnet": {
        "description": "WordNet lexical database",
        "required": False
    },
    "averaged_perceptron_tagger": {
        "description": "Averaged perceptron tagger",
        "required": False
    }
}

CUSTOM_MODELS = {
    "skills_classifier": {
        "url": "https://example.com/models/skills_classifier.pkl",
        "filename": "skills_classifier.pkl",
        "description": "Custom skills classification model",
        "required": False
    }
}


class ModelDownloader:
    """Model download and management class."""
    
    def __init__(self, models_dir: str = "models"):
        """
        Initialize model downloader.
        
        Args:
            models_dir: Directory to store downloaded models
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.models_dir / "spacy").mkdir(exist_ok=True)
        (self.models_dir / "nltk").mkdir(exist_ok=True)
        (self.models_dir / "custom").mkdir(exist_ok=True)
        
        self.downloaded_models = self._load_download_log()
    
    def _load_download_log(self) -> Dict[str, Any]:
        """Load download log to track installed models."""
        log_file = self.models_dir / "download_log.json"
        
        if log_file.exists():
            try:
                with open(log_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load download log: {e}")
        
        return {}
    
    def _save_download_log(self):
        """Save download log."""
        log_file = self.models_dir / "download_log.json"
        
        try:
            with open(log_file, 'w') as f:
                json.dump(self.downloaded_models, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save download log: {e}")
    
    def _run_command(self, command: List[str]) -> bool:
        """
        Run shell command and return success status.
        
        Args:
            command: Command to run as list of strings
            
        Returns:
            bool: True if command succeeded
        """
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True
            )
            logger.debug(f"Command output: {result.stdout}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed: {' '.join(command)}")
            logger.error(f"Error: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error running command: {e}")
            return False
    
    def check_spacy_installation(self) -> bool:
        """Check if spaCy is installed."""
        try:
            import spacy
            logger.info(f"spaCy version {spacy.__version__} is installed")
            return True
        except ImportError:
            logger.error("spaCy is not installed")
            return False
    
    def download_spacy_models(self, models: Optional[List[str]] = None) -> bool:
        """
        Download spaCy models.
        
        Args:
            models: List of model names to download. If None, downloads required models.
            
        Returns:
            bool: True if all downloads succeeded
        """
        if not self.check_spacy_installation():
            logger.error("Please install spaCy first: pip install spacy")
            return False
        
        if models is None:
            models = [name for name, config in SPACY_MODELS.items() if config["required"]]
        
        success = True
        
        for model_name in models:
            if model_name not in SPACY_MODELS:
                logger.warning(f"Unknown spaCy model: {model_name}")
                continue
            
            model_config = SPACY_MODELS[model_name]
            
            # Check if already downloaded
            if self._is_spacy_model_installed(model_name):
                logger.info(f"spaCy model '{model_name}' is already installed")
                continue
            
            logger.info(f"Downloading spaCy model: {model_name}")
            logger.info(f"Description: {model_config['description']}")
            logger.info(f"Size: {model_config['size']}")
            
            # Download using spacy download command
            command = [sys.executable, "-m", "spacy", "download", model_name]
            
            if self._run_command(command):
                logger.info(f"Successfully downloaded spaCy model: {model_name}")
                self.downloaded_models[f"spacy_{model_name}"] = {
                    "type": "spacy",
                    "name": model_name,
                    "version": model_config["version"],
                    "downloaded_at": datetime.now().isoformat()
                }
            else:
                logger.error(f"Failed to download spaCy model: {model_name}")
                success = False
        
        return success
    
    def _is_spacy_model_installed(self, model_name: str) -> bool:
        """Check if spaCy model is installed."""
        try:
            import spacy
            spacy.load(model_name)
            return True
        except (ImportError, OSError):
            return False
    
    def download_nltk_data(self, datasets: Optional[List[str]] = None) -> bool:
        """
        Download NLTK data.
        
        Args:
            datasets: List of NLTK datasets to download
            
        Returns:
            bool: True if all downloads succeeded
        """
        try:
            import nltk
        except ImportError:
            logger.error("NLTK is not installed. Please install: pip install nltk")
            return False
        
        if datasets is None:
            datasets = [name for name, config in NLTK_DATA.items() if config["required"]]
        
        success = True
        
        # Set NLTK data path
        nltk_data_dir = self.models_dir / "nltk"
        nltk.data.path.append(str(nltk_data_dir))
        
        for dataset_name in datasets:
            if dataset_name not in NLTK_DATA:
                logger.warning(f"Unknown NLTK dataset: {dataset_name}")
                continue
            
            dataset_config = NLTK_DATA[dataset_name]
            
            logger.info(f"Downloading NLTK dataset: {dataset_name}")
            logger.info(f"Description: {dataset_config['description']}")
            
            try:
                nltk.download(dataset_name, download_dir=str(nltk_data_dir))
                logger.info(f"Successfully downloaded NLTK dataset: {dataset_name}")
                
                self.downloaded_models[f"nltk_{dataset_name}"] = {
                    "type": "nltk",
                    "name": dataset_name,
                    "downloaded_at": datetime.now().isoformat()
                }
            except Exception as e:
                logger.error(f"Failed to download NLTK dataset {dataset_name}: {e}")
                success = False
        
        return success
    
    def download_custom_models(self, models: Optional[List[str]] = None) -> bool:
        """
        Download custom models.
        
        Args:
            models: List of custom model names to download
            
        Returns:
            bool: True if all downloads succeeded
        """
        if models is None:
            models = [name for name, config in CUSTOM_MODELS.items() if config["required"]]
        
        success = True
        
        for model_name in models:
            if model_name not in CUSTOM_MODELS:
                logger.warning(f"Unknown custom model: {model_name}")
                continue
            
            model_config = CUSTOM_MODELS[model_name]
            model_path = self.models_dir / "custom" / model_config["filename"]
            
            # Check if already downloaded
            if model_path.exists():
                logger.info(f"Custom model '{model_name}' already exists")
                continue
            
            logger.info(f"Downloading custom model: {model_name}")
            logger.info(f"Description: {model_config['description']}")
            
            try:
                urllib.request.urlretrieve(model_config["url"], str(model_path))
                logger.info(f"Successfully downloaded custom model: {model_name}")
                
                self.downloaded_models[f"custom_{model_name}"] = {
                    "type": "custom",
                    "name": model_name,
                    "filename": model_config["filename"],
                    "downloaded_at": datetime.now().isoformat()
                }
            except Exception as e:
                logger.error(f"Failed to download custom model {model_name}: {e}")
                success = False
        
        return success
    
    def verify_models(self) -> Dict[str, bool]:
        """
        Verify that all downloaded models are working.
        
        Returns:
            Dict mapping model names to verification status
        """
        verification_results = {}
        
        # Verify spaCy models
        for model_name in SPACY_MODELS:
            try:
                import spacy
                nlp = spacy.load(model_name)
                # Test with sample text
                doc = nlp("This is a test sentence.")
                verification_results[f"spacy_{model_name}"] = len(doc) > 0
                logger.info(f"spaCy model '{model_name}' verification: PASSED")
            except Exception as e:
                verification_results[f"spacy_{model_name}"] = False
                logger.error(f"spaCy model '{model_name}' verification: FAILED - {e}")
        
        # Verify NLTK data
        for dataset_name in NLTK_DATA:
            try:
                import nltk
                # Try to access the dataset
                if dataset_name == "punkt":
                    nltk.data.find('tokenizers/punkt')
                elif dataset_name == "stopwords":
                    nltk.data.find('corpora/stopwords')
                elif dataset_name == "wordnet":
                    nltk.data.find('corpora/wordnet')
                
                verification_results[f"nltk_{dataset_name}"] = True
                logger.info(f"NLTK dataset '{dataset_name}' verification: PASSED")
            except Exception as e:
                verification_results[f"nltk_{dataset_name}"] = False
                logger.error(f"NLTK dataset '{dataset_name}' verification: FAILED - {e}")
        
        # Verify custom models
        for model_name, config in CUSTOM_MODELS.items():
            model_path = self.models_dir / "custom" / config["filename"]
            verification_results[f"custom_{model_name}"] = model_path.exists()
            
            if model_path.exists():
                logger.info(f"Custom model '{model_name}' verification: PASSED")
            else:
                logger.error(f"Custom model '{model_name}' verification: FAILED - File not found")
        
        return verification_results
    
    def list_available_models(self):
        """List all available models."""
        print("\n" + "="*60)
        print("Available Models for Download")
        print("="*60)
        
        print("\nspaCy Models:")
        print("-" * 40)
        for name, config in SPACY_MODELS.items():
            status = "✓ Installed" if self._is_spacy_model_installed(name) else "✗ Not installed"
            required = "Required" if config["required"] else "Optional"
            print(f"  {name:<20} | {status:<15} | {required:<10} | {config['size']}")
            print(f"    {config['description']}")
        
        print("\nNLTK Datasets:")
        print("-" * 40)
        for name, config in NLTK_DATA.items():
            required = "Required" if config["required"] else "Optional"
            print(f"  {name:<20} | {required:<10}")
            print(f"    {config['description']}")
        
        print("\nCustom Models:")
        print("-" * 40)
        for name, config in CUSTOM_MODELS.items():
            model_path = self.models_dir / "custom" / config["filename"]
            status = "✓ Downloaded" if model_path.exists() else "✗ Not downloaded"
            required = "Required" if config["required"] else "Optional"
            print(f"  {name:<20} | {status:<15} | {required:<10}")
            print(f"    {config['description']}")
        
        print("\n" + "="*60)
    
    def cleanup_models(self):
        """Clean up downloaded models."""
        logger.info("Cleaning up model directories...")
        
        try:
            import shutil
            if self.models_dir.exists():
                shutil.rmtree(self.models_dir)
                self.models_dir.mkdir(exist_ok=True)
                logger.info("Model directories cleaned up successfully")
        except Exception as e:
            logger.error(f"Failed to cleanup models: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about downloaded models."""
        info = {
            "models_directory": str(self.models_dir),
            "total_models": len(self.downloaded_models),
            "models_by_type": {},
            "disk_usage": self._get_disk_usage(),
            "last_update": datetime.now().isoformat()
        }
        
        # Group by type
        for model_id, model_data in self.downloaded_models.items():
            model_type = model_data["type"]
            if model_type not in info["models_by_type"]:
                info["models_by_type"][model_type] = []
            info["models_by_type"][model_type].append(model_data)
        
        return info
    
    def _get_disk_usage(self) -> str:
        """Get disk usage of models directory."""
        try:
            import shutil
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(self.models_dir):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    total_size += os.path.getsize(filepath)
            
            # Convert to human readable format
            for unit in ['B', 'KB', 'MB', 'GB']:
                if total_size < 1024.0:
                    return f"{total_size:.1f} {unit}"
                total_size /= 1024.0
            return f"{total_size:.1f} TB"
        except Exception:
            return "Unknown"


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Download and manage ML models for AI Resume Screener"
    )
    
    parser.add_argument(
        "--spacy-models",
        nargs="*",
        help="spaCy models to download (default: required models only)"
    )
    
    parser.add_argument(
        "--nltk-data",
        nargs="*", 
        help="NLTK datasets to download (default: required datasets only)"
    )
    
    parser.add_argument(
        "--custom-models",
        nargs="*",
        help="Custom models to download"
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all available models"
    )
    
    parser.add_argument(
        "--required-only",
        action="store_true",
        help="Download only required models (default)"
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models"
    )
    
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify downloaded models"
    )
    
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show model information"
    )
    
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Clean up all downloaded models"
    )
    
    parser.add_argument(
        "--models-dir",
        default="models",
        help="Directory to store models (default: models)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create model downloader
    downloader = ModelDownloader(args.models_dir)
    
    # Handle different commands
    if args.list:
        downloader.list_available_models()
        return
    
    if args.info:
        info = downloader.get_model_info()
        print(json.dumps(info, indent=2))
        return
    
    if args.verify:
        results = downloader.verify_models()
        print("\nModel Verification Results:")
        print("-" * 40)
        for model, status in results.items():
            status_str = "✓ PASSED" if status else "✗ FAILED"
            print(f"  {model:<30} | {status_str}")
        return
    
    if args.cleanup:
        downloader.cleanup_models()
        return
    
    # Download models
    success = True
    
    if args.all:
        # Download all models
        logger.info("Downloading all available models...")
        success &= downloader.download_spacy_models(list(SPACY_MODELS.keys()))
        success &= downloader.download_nltk_data(list(NLTK_DATA.keys()))
        success &= downloader.download_custom_models(list(CUSTOM_MODELS.keys()))
    else:
        # Download specified or required models
        if args.spacy_models is not None:
            success &= downloader.download_spacy_models(args.spacy_models)
        elif not args.nltk_data and not args.custom_models:
            # Default: download required spaCy models
            success &= downloader.download_spacy_models()
        
        if args.nltk_data is not None:
            success &= downloader.download_nltk_data(args.nltk_data)
        
        if args.custom_models is not None:
            success &= downloader.download_custom_models(args.custom_models)
    
    # Save download log
    downloader._save_download_log()
    
    if success:
        logger.info("All model downloads completed successfully!")
        
        # Run verification
        logger.info("Running model verification...")
        results = downloader.verify_models()
        failed_models = [model for model, status in results.items() if not status]
        
        if failed_models:
            logger.warning(f"Some models failed verification: {failed_models}")
        else:
            logger.info("All models verified successfully!")
    else:
        logger.error("Some model downloads failed. Check the logs for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
