"""
Configuration Loader

Loads environment variables from .env file.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Any

# Load .env file
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)


class Config:
    """Configuration from environment variables."""
    
    # ========== AIRFLOW ==========
    AIRFLOW_UID = os.getenv('AIRFLOW_UID', '50000')
    
    # ========== API KEYS ==========
    ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY', '')
    FRED_API_KEY = os.getenv('FRED_API_KEY', '')
    FMP_API_KEY = os.getenv('FMP_API_KEY', 'demo')
    
    # ========== SLACK ==========
    SLACK_WEBHOOK_URL = os.getenv('SLACK_WEBHOOK_URL', '')
    SLACK_ALERTS_ENABLED = os.getenv('SLACK_ALERTS_ENABLED', 'false').lower() == 'true'
    
    # ========== EMAIL ==========
    EMAIL_ALERTS_ENABLED = os.getenv('EMAIL_ALERTS_ENABLED', 'false').lower() == 'true'
    SMTP_SERVER = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
    SMTP_PORT = int(os.getenv('SMTP_PORT', '587'))
    SENDER_EMAIL = os.getenv('SENDER_EMAIL', '')
    SENDER_APP_PASSWORD = os.getenv('SENDER_APP_PASSWORD', '')
    RECIPIENT_EMAILS = os.getenv('RECIPIENT_EMAILS', '').split(',')
    
    # ========== DVC ==========
    DVC_REMOTE_TYPE = os.getenv('DVC_REMOTE_TYPE', 'local')
    DVC_LOCAL_REMOTE = os.getenv('DVC_LOCAL_REMOTE', '/tmp/dvc-storage')
    
    # AWS
    AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID', '')
    AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY', '')
    DVC_S3_BUCKET = os.getenv('DVC_S3_BUCKET', '')
    
    # ========== PIPELINE PARAMETERS ==========
    START_DATE = os.getenv('START_DATE', '1990-01-01')
    END_DATE = os.getenv('END_DATE', 'today')
    REPORTING_LAG_DAYS = int(os.getenv('REPORTING_LAG_DAYS', '45'))
    
    # ========== THRESHOLDS ==========
    ANOMALY_IQR_THRESHOLD = float(os.getenv('ANOMALY_IQR_THRESHOLD', '3.0'))
    BIAS_REPRESENTATION_THRESHOLD = float(os.getenv('BIAS_REPRESENTATION_THRESHOLD', '0.3'))
    DRIFT_SIGNIFICANCE_LEVEL = float(os.getenv('DRIFT_SIGNIFICANCE_LEVEL', '0.05'))
    
    # ========== LOGGING ==========
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_DIR = os.getenv('LOG_DIR', 'logs')
    
    # ========== VALIDATION ==========
    MAX_MISSING_PCT_CLEAN = float(os.getenv('MAX_MISSING_PCT_CLEAN', '5'))
    MAX_MISSING_PCT_MERGED = float(os.getenv('MAX_MISSING_PCT_MERGED', '2'))
    
    @classmethod
    def get_all(cls) -> Dict[str, Any]:
        """Get all configuration as dictionary."""
        return {
            key: value for key, value in cls.__dict__.items()
            if not key.startswith('_') and not callable(value)
        }
    
    @classmethod
    def validate(cls):
        """Validate that required configs are set."""
        required = [
            'ALPHA_VANTAGE_API_KEY'
        ]
        
        missing = []
        for key in required:
            if not getattr(cls, key):
                missing.append(key)
        
        if missing:
            raise ValueError(f"Missing required environment variables: {missing}")
        
        return True


# Validate on import
if __name__ != "__main__":
    try:
        Config.validate()
    except ValueError as e:
        print(f"⚠️  Warning: {e}")