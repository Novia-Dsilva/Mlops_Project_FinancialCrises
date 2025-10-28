"""
DVC Helper Functions

Utilities for DVC data versioning.
"""

import subprocess
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def dvc_add_and_push(data_path: str):
    """
    Add data to DVC and push to remote.
    
    Args:
        data_path: Path to data file or directory
    """
    logger.info(f"📦 Versioning {data_path} with DVC...")
    
    try:
        # Add to DVC
        result = subprocess.run(
            ['dvc', 'add', data_path],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            logger.error(f"   ✗ DVC add failed: {result.stderr}")
            return False
        
        logger.info(f"   ✓ Added to DVC")
        
        # Push to remote
        result = subprocess.run(
            ['dvc', 'push'],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            logger.warning(f"   ⚠️  DVC push failed: {result.stderr}")
            return False
        
        logger.info(f"   ✓ Pushed to remote")
        
        return True
        
    except Exception as e:
        logger.error(f"   ✗ DVC error: {e}")
        return False


def dvc_pull(data_path: str = None):
    """
    Pull data from DVC remote.
    
    Args:
        data_path: Specific path to pull (None = pull all)
    """
    logger.info(f"📥 Pulling from DVC remote...")
    
    try:
        cmd = ['dvc', 'pull']
        if data_path:
            cmd.append(data_path)
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"   ✗ DVC pull failed: {result.stderr}")
            return False
        
        logger.info(f"   ✓ Pulled from remote")
        return True
        
    except Exception as e:
        logger.error(f"   ✗ DVC error: {e}")
        return False