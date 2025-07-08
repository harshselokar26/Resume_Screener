#!/usr/bin/env python3
"""
Download spaCy models for AI Resume Screener
"""

import subprocess
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_spacy_model(model_name="en_core_web_sm"):
    """Download spaCy model."""
    try:
        logger.info(f"Downloading spaCy model: {model_name}")
        subprocess.check_call([sys.executable, "-m", "spacy", "download", model_name])
        logger.info(f"Successfully downloaded {model_name}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to download {model_name}: {e}")
        return False

if __name__ == "__main__":
    success = download_spacy_model()
    if success:
        print("✅ spaCy models downloaded successfully")
    else:
        print("❌ Failed to download spaCy models")
        sys.exit(1)
