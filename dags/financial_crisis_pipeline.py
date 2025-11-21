"""
Financial Crisis Detection Pipeline - Clean & Modular DAG with Alerting
========================================================================
Includes all 4 validation checkpoints + uses existing alerting system.
 
Author: MLOps Group11 Team
"""

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import os
import sys

# Add project to path for imports
PROJECT_DIR = '/opt/airflow/project'
sys.path.insert(0, PROJECT_DIR)

# Import your existing alerting system
try:
    from src.monitoring.alerting import AlertManager
    ALERTING_AVAILABLE = True
except ImportError:
    print("WARNING: AlertManager not available. Alerts disabled.")
    ALERTING_AVAILABLE = False

# =======================================================================
# CONFIGURATION
# =======================================================================

# Pipeline steps configuration
PIPELINE_STEPS = [
    ('step0_collect_data', 'src/data/step0_data_collection.py',
     'Collect data from APIs', 90, True),
    ('validate_checkpoint_1', 'src/validation/validate_checkpoint_1_raw.py',
     'Validate raw data', 10, True),
    ('step1_clean_data', 'src/data/step1_data_cleaning.py',
     'Clean data (PIT correct)', 20, True),
    ('validate_checkpoint_2', 'src/validation/validate_checkpoint_2_clean.py',
     'Validate clean data', 10, True),
    ('step2_engineer_features', 'src/data/step2_feature_engineering.py',
     'Engineer features', 20, False),
    ('step3_merge_data', 'src/data/step3_data_merging.py', 'Merge datasets', 15, True),
    ('step4_clean_merged', 'src/data/step4_post_merge_cleaning.py',
     'Clean merged data', 10, False),
    ('validate_checkpoint_3', 'src/validation/validate_checkpoint_3_merged.py',
     'Validate merged data', 10, True),
    ('step5_detect_bias', 'src/data/step5_bias_detection_with_explicit_slicing.py',
     'Detect bias', 10, False),
    ('step6_detect_anomalies', 'src/data/step6_anomaly_detection.py',
     'Detect anomalies', 10, False),
    ('step7_detect_drift', 'src/data/step7_drift_detection.py',
     'Detect drift', 10, False),
]

# =======================================================================
# ALERTING CALLBACKS
# =======================================================================


def task_failure_alert(context):
    if not ALERTING_AVAILABLE:
        return
    try:
        task = context.get('task_instance')
        dag_run = context.get('dag_run')
        exception = context.get('exception')

        alert_manager = AlertManager()
        is_validation = 'validate' in task.task_id
        is_critical = any(step[0] == task.task_id and step[4]
                          for step in PIPELINE_STEPS)
        severity = 'CRITICAL' if (is_validation or is_critical) else 'ERROR'

        message = f"""
        Pipeline Task Failed: {task.task_id}
        DAG: {task.dag_id}
        Execution Date: {dag_run.execution_date}
        Task: {task.task_id}
        Error: {str(exception) if exception else 'Check logs for details'}
        Log URL: {task.log_url}
        """
        if is_validation:
            message += "\nWARNING: Data Validation Failed - Pipeline stopped to prevent bad data propagation."

        alert_manager.send_alert(
            message=message,
            severity=severity,
            component=task.task_id,
            alert_type='PIPELINE_FAILURE'
        )
        print(f"Alert sent for {task.task_id} failure")
    except Exception as e:
        print(f"Failed to send alert: {str(e)}")


def pipeline_success_alert(**context):
    if not ALERTING_AVAILABLE:
        return
    try:
        dag_run = context.get('dag_run')
        duration = dag_run.end_date - dag_run.start_date if dag_run.end_date else "N/A"
        alert_manager = AlertManager()

        message = f"""
        SUCCESS: Financial Crisis Pipeline Completed
        Execution Date: {dag_run.execution_date}
        Duration: {duration}
        
        Pipeline Summary:
        - Data collected & validated (Checkpoint 1)
        - Data cleaned & validated (Checkpoint 2)
        - Features engineered
        - Data merged & validated (Checkpoint 3)
        - Post-merge cleaning & validation (Checkpoint 4)
        - Anomaly detection completed
        - Bias detection completed
        - Drift detection completed
        
        Data ready for model training!
        """

        alert_manager.send_alert(
            message=message,
            severity='INFO',
            component='pipeline',
            alert_type='PIPELINE_SUCCESS'
        )
        print("Success alert sent")
    except Exception as e:
        print(f"Failed to send success alert: {str(e)}")


def validation_failure_alert(context):
    if not ALERTING_AVAILABLE:
        return
    try:
        task = context.get('task_instance')
        alert_manager = AlertManager()

        checkpoint_num = '1' if 'checkpoint_1' in task.task_id else \
            '2' if 'checkpoint_2' in task.task_id else \
            '3' if 'checkpoint_3' in task.task_id else '4'

        message = f"""
        CRITICAL: Validation Checkpoint {checkpoint_num} Failed
        Task: {task.task_id}
        Execution Date: {context.get('dag_run').execution_date}
        
        Data quality issues detected. Pipeline has been stopped.
        
        Action Required:
        1. Check validation report: data/validation_reports/
        2. Review failed expectations in Great Expectations
        3. Fix data quality issues
        4. Re-run pipeline
        
        Log URL: {task.log_url}
        """

        alert_manager.send_alert(
            message=message,
            severity='CRITICAL',
            component=f'validation_checkpoint_{checkpoint_num}',
            alert_type='VALIDATION_FAILURE'
        )
        print(
            f"CRITICAL validation alert sent for checkpoint {checkpoint_num}")
    except Exception as e:
        print(f"Failed to send validation alert: {str(e)}")


# =======================================================================
# DAG DEFAULT ARGS
# =======================================================================

default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'on_failure_callback': task_failure_alert,
}


# =======================================================================
# DAG DEFINITION
# =======================================================================

with DAG(
    'financial_crisis_pipeline',
    default_args=default_args,
    description='Financial Crisis Pipeline with Validation & Custom Alerting',
    schedule_interval=None,  # Manual trigger
    start_date=days_ago(1),
    catchup=False,
    tags=['mlops', 'financial', 'validation', 'alerting'],
    max_active_runs=1,
) as dag:

    # Create tasks dynamically
    tasks = {}

    for task_id, script, desc, timeout, critical in PIPELINE_STEPS:
        callbacks = {}
        if 'validate' in task_id:
            callbacks['on_failure_callback'] = validation_failure_alert
        else:
            callbacks['on_failure_callback'] = task_failure_alert

        tasks[task_id] = BashOperator(
            task_id=task_id,
            bash_command=f"""
            cd {PROJECT_DIR} && \
            echo "{'Validating' if 'validate' in task_id else 'Running'}: {desc}..." && \
            python {script}
            """,
            execution_timeout=timedelta(minutes=timeout),
            **callbacks
        )

    # Success notification task
    pipeline_success = PythonOperator(
        task_id='pipeline_success',
        python_callable=pipeline_success_alert,
        trigger_rule='all_success'
    )

    # Chain all pipeline steps
    for i in range(len(PIPELINE_STEPS) - 1):
        current_task = PIPELINE_STEPS[i][0]
        next_task = PIPELINE_STEPS[i + 1][0]
        tasks[current_task] >> tasks[next_task]

    tasks[PIPELINE_STEPS[-1][0]] >> pipeline_success
