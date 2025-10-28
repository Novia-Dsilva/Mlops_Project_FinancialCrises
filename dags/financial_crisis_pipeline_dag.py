"""
Financial Crisis Detection - Complete MLOps Pipeline DAG

This DAG orchestrates the complete data pipeline:
- Data collection from APIs
- Data validation at each stage
- Data cleaning and preprocessing
- Feature engineering
- Anomaly detection
- Bias detection
- Drift detection
- DVC versioning
- Monitoring and alerting

Author: MLOps Team
Schedule: Daily at 2 AM
"""

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.email import EmailOperator
from airflow.utils.dates import days_ago
from airflow.utils.task_group import TaskGroup
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path
project_root = Path('/opt/airflow')  # Adjust to your Airflow setup
sys.path.insert(0, str(project_root))

# Import pipeline modules
from src.monitoring.alerting import send_slack_alert, send_email_alert
from src.utils.dvc_helper import dvc_add_and_push, dvc_pull


# ============================================================================
# DAG DEFAULT ARGUMENTS
# ============================================================================

default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'email': ['your-email@example.com'],  # ← UPDATE THIS
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=2),
}


# ============================================================================
# TASK FUNCTIONS
# ============================================================================

def task_data_collection(**context):
    """Task: Collect raw data from APIs."""
    import subprocess
    
    result = subprocess.run(
        ['python', 'step0_data_collection.py'],
        capture_output=True,
        text=True,
        cwd=project_root
    )
    
    if result.returncode != 0:
        raise Exception(f"Data collection failed: {result.stderr}")
    
    # Push success metric
    context['task_instance'].xcom_push(key='data_collection_status', value='SUCCESS')
    
    return "Data collection completed"


def task_validate_raw(**context):
    """Task: Validate raw data (Checkpoint 1)."""
    import subprocess
    
    result = subprocess.run(
        ['python', 'src/validation/validate_checkpoint_1_raw.py'],
        capture_output=True,
        text=True,
        cwd=project_root
    )
    
    if result.returncode != 0:
        # Send alert
        send_slack_alert(
            "🚨 CHECKPOINT 1 FAILED: Raw data validation failed",
            result.stderr
        )
        raise Exception(f"Raw data validation failed: {result.stderr}")
    
    return "Raw data validated"


def task_data_cleaning(**context):
    """Task: Clean data."""
    import subprocess
    
    result = subprocess.run(
        ['python', 'step1_data_cleaning.py'],
        capture_output=True,
        text=True,
        cwd=project_root
    )
    
    if result.returncode != 0:
        raise Exception(f"Data cleaning failed: {result.stderr}")
    
    return "Data cleaned"


def task_validate_cleaned(**context):
    """Task: Validate cleaned data (Checkpoint 2)."""
    import subprocess
    
    result = subprocess.run(
        ['python', 'src/validation/validate_checkpoint_2_clean.py'],
        capture_output=True,
        text=True,
        cwd=project_root
    )
    
    if result.returncode != 0:
        send_slack_alert(
            "🚨 CHECKPOINT 2 FAILED: Cleaned data validation failed",
            result.stderr
        )
        raise Exception(f"Cleaned data validation failed: {result.stderr}")
    
    return "Cleaned data validated"


def task_feature_engineering(**context):
    """Task: Engineer features."""
    import subprocess
    
    result = subprocess.run(
        ['python', 'step2_feature_engineering.py'],
        capture_output=True,
        text=True,
        cwd=project_root
    )
    
    if result.returncode != 0:
        raise Exception(f"Feature engineering failed: {result.stderr}")
    
    return "Features engineered"


def task_data_merging(**context):
    """Task: Merge datasets."""
    import subprocess
    
    result = subprocess.run(
        ['python', 'step3_data_merging.py'],
        capture_output=True,
        text=True,
        cwd=project_root
    )
    
    if result.returncode != 0:
        raise Exception(f"Data merging failed: {result.stderr}")
    
    return "Data merged"


def task_validate_merged(**context):
    """Task: Validate merged data (Checkpoint 3)."""
    import subprocess
    
    result = subprocess.run(
        ['python', 'src/validation/validate_checkpoint_3_merged.py'],
        capture_output=True,
        text=True,
        cwd=project_root
    )
    
    if result.returncode != 0:
        send_slack_alert(
            "🚨 CHECKPOINT 3 FAILED: Merged data validation failed",
            result.stderr
        )
        raise Exception(f"Merged data validation failed: {result.stderr}")
    
    return "Merged data validated"


def task_clean_merged(**context):
    """Task: Clean merged data."""
    import subprocess
    
    result = subprocess.run(
        ['python', 'step3c_post_merge_cleaning.py'],
        capture_output=True,
        text=True,
        cwd=project_root
    )
    
    if result.returncode != 0:
        raise Exception(f"Post-merge cleaning failed: {result.stderr}")
    
    return "Merged data cleaned"


def task_anomaly_detection(**context):
    """Task: Detect anomalies."""
    import subprocess
    
    result = subprocess.run(
        ['python', 'src/validation/step5_anomaly_detection.py'],
        capture_output=True,
        text=True,
        cwd=project_root
    )
    
    # Check for critical anomalies in output
    if "CRITICAL" in result.stdout:
        send_slack_alert(
            "⚠️ CRITICAL ANOMALIES DETECTED",
            "Check anomaly report in data/anomaly_reports/"
        )
    
    if result.returncode != 0:
        raise Exception(f"Anomaly detection failed: {result.stderr}")
    
    return "Anomalies detected and flagged"


def task_bias_detection(**context):
    """Task: Detect bias through data slicing."""
    import subprocess
    
    result = subprocess.run(
        ['python', 'step4_bias_detection.py'],
        capture_output=True,
        text=True,
        cwd=project_root
    )
    
    if result.returncode != 0:
        raise Exception(f"Bias detection failed: {result.stderr}")
    
    return "Bias detection complete"


def task_drift_detection(**context):
    """Task: Detect data drift."""
    import subprocess
    
    result = subprocess.run(
        ['python', 'step6_drift_detection.py'],
        capture_output=True,
        text=True,
        cwd=project_root
    )
    
    # Check for high drift
    if "HIGH drift" in result.stdout:
        send_slack_alert(
            "⚠️ HIGH DRIFT DETECTED",
            "Multiple features show significant drift. Review drift report."
        )
    
    if result.returncode != 0:
        raise Exception(f"Drift detection failed: {result.stderr}")
    
    return "Drift detection complete"


def task_dvc_version_data(**context):
    """Task: Version data with DVC."""
    from src.utils.dvc_helper import dvc_add_and_push
    
    # Version all data stages
    datasets = [
        'data/raw',
        'data/clean',
        'data/features'
    ]
    
    for dataset in datasets:
        dvc_add_and_push(dataset)
    
    return "Data versioned with DVC"


def task_generate_statistics(**context):
    """Task: Generate data statistics and schema."""
    import subprocess
    
    # Generate Great Expectations data docs
    result = subprocess.run(
        ['great_expectations', 'docs', 'build'],
        capture_output=True,
        text=True,
        cwd=project_root
    )
    
    return "Statistics and schema generated"


def task_send_success_notification(**context):
    """Task: Send success notification."""
    
    message = f"""
    ✅ PIPELINE COMPLETED SUCCESSFULLY
    
    Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    
    All steps completed:
    ✓ Data collection
    ✓ Data cleaning  
    ✓ Feature engineering
    ✓ Data merging
    ✓ Anomaly detection
    ✓ Bias detection
    ✓ Drift detection
    ✓ DVC versioning
    
    Ready for model training!
    """
    
    send_slack_alert("✅ Pipeline Success", message)
    
    return "Success notification sent"


# ============================================================================
# CREATE DAG
# ============================================================================

with DAG(
    'financial_crisis_detection_pipeline',
    default_args=default_args,
    description='Complete MLOps pipeline for financial crisis detection',
    schedule_interval='0 2 * * *',  # Daily at 2 AM
    start_date=days_ago(1),
    catchup=False,
    tags=['mlops', 'financial', 'data-pipeline'],
) as dag:
    
    # ========================================================================
    # TASK: START
    # ========================================================================
    
    start = BashOperator(
        task_id='start_pipeline',
        bash_command='echo "Starting Financial Crisis Detection Pipeline"'
    )
    
    # ========================================================================
    # TASK GROUP: DATA ACQUISITION
    # ========================================================================
    
    with TaskGroup('data_acquisition') as data_acquisition:
        
        collect_data = PythonOperator(
            task_id='collect_raw_data',
            python_callable=task_data_collection,
            provide_context=True
        )
        
        validate_raw = PythonOperator(
            task_id='validate_raw_data',
            python_callable=task_validate_raw,
            provide_context=True
        )
        
        version_raw = BashOperator(
            task_id='version_raw_with_dvc',
            bash_command='dvc add data/raw && dvc push'
        )
        
        collect_data >> validate_raw >> version_raw
    
    # ========================================================================
    # TASK GROUP: DATA PREPROCESSING
    # ========================================================================
    
    with TaskGroup('data_preprocessing') as data_preprocessing:
        
        clean_data = PythonOperator(
            task_id='clean_data',
            python_callable=task_data_cleaning,
            provide_context=True
        )
        
        validate_cleaned = PythonOperator(
            task_id='validate_cleaned_data',
            python_callable=task_validate_cleaned,
            provide_context=True
        )
        
        version_clean = BashOperator(
            task_id='version_clean_with_dvc',
            bash_command='dvc add data/clean && dvc push'
        )
        
        clean_data >> validate_cleaned >> version_clean
    
    # ========================================================================
    # TASK GROUP: FEATURE ENGINEERING
    # ========================================================================
    
    with TaskGroup('feature_engineering') as feature_engineering:
        
        engineer_features = PythonOperator(
            task_id='engineer_features',
            python_callable=task_feature_engineering,
            provide_context=True
        )
        
        merge_data = PythonOperator(
            task_id='merge_datasets',
            python_callable=task_data_merging,
            provide_context=True
        )
        
        validate_merged = PythonOperator(
            task_id='validate_merged_data',
            python_callable=task_validate_merged,
            provide_context=True
        )
        
        clean_merged = PythonOperator(
            task_id='clean_merged_data',
            python_callable=task_clean_merged,
            provide_context=True
        )
        
        version_features = BashOperator(
            task_id='version_features_with_dvc',
            bash_command='dvc add data/features && dvc push'
        )
        
        engineer_features >> merge_data >> validate_merged >> clean_merged >> version_features
    
    # ========================================================================
    # TASK GROUP: DATA QUALITY
    # ========================================================================
    
    with TaskGroup('data_quality') as data_quality:
        
        detect_anomalies = PythonOperator(
            task_id='detect_anomalies',
            python_callable=task_anomaly_detection,
            provide_context=True
        )
        
        detect_bias = PythonOperator(
            task_id='detect_bias',
            python_callable=task_bias_detection,
            provide_context=True
        )
        
        detect_drift = PythonOperator(
            task_id='detect_drift',
            python_callable=task_drift_detection,
            provide_context=True
        )
        
        generate_stats = PythonOperator(
            task_id='generate_statistics',
            python_callable=task_generate_statistics,
            provide_context=True
        )
        
        # Run in parallel
        [detect_anomalies, detect_bias, detect_drift] >> generate_stats
    
    # ========================================================================
    # TASK: FINAL DVC COMMIT
    # ========================================================================
    
    final_dvc_commit = BashOperator(
        task_id='final_dvc_commit',
        bash_command='''
        git add data/*.dvc .dvc/config
        git commit -m "Pipeline run $(date +%Y%m%d_%H%M%S)" || true
        git push || true
        dvc push
        '''
    )
    
    # ========================================================================
    # TASK: SUCCESS NOTIFICATION
    # ========================================================================
    
    send_success = PythonOperator(
        task_id='send_success_notification',
        python_callable=task_send_success_notification,
        provide_context=True
    )
    
    # ========================================================================
    # TASK: END
    # ========================================================================
    
    end = BashOperator(
        task_id='end_pipeline',
        bash_command='echo "Pipeline completed successfully"'
    )
    
    # ========================================================================
    # DAG FLOW
    # ========================================================================
    
    start >> data_acquisition >> data_preprocessing >> feature_engineering >> data_quality >> final_dvc_commit >> send_success >> end