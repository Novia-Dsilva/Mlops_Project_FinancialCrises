# Recreate the DAG file cleanly

"""
Financial Crisis Detection Pipeline - Airflow DAG
"""

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from datetime import timedelta

PROJECT_DIR = '/opt/airflow/project'

default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'email_on_failure': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'financial_crisis_pipeline',
    default_args=default_args,
    description='Financial Crisis Detection Pipeline',
    schedule_interval=None,
    start_date=days_ago(1),
    catchup=False,
    tags=['mlops', 'financial'],
) as dag:
    
    start = BashOperator(
        task_id='start',
        bash_command='echo "Starting Pipeline"'
    )
    
    collect_data = BashOperator(
        task_id='step0_collect_data',
        bash_command=f'cd {PROJECT_DIR} && python src/data/step0_data_collection.py',
        execution_timeout=timedelta(minutes=30)
    )
    
    clean_data = BashOperator(
        task_id='step1_clean_data',
        bash_command=f'cd {PROJECT_DIR} && python src/data/step1_data_cleaning.py',
        execution_timeout=timedelta(minutes=20)
    )
    
    engineer_features = BashOperator(
        task_id='step2_engineer_features',
        bash_command=f'cd {PROJECT_DIR} && python src/data/step2_feature_engineering.py',
        execution_timeout=timedelta(minutes=20)
    )
    
    merge_data = BashOperator(
        task_id='step3_merge_data',
        bash_command=f'cd {PROJECT_DIR} && python src/data/step3_data_merging.py',
        execution_timeout=timedelta(minutes=15)
    )
    
    clean_merged = BashOperator(
        task_id='step4_clean_merged',
        bash_command=f'cd {PROJECT_DIR} && python src/data/step4_post_merge_cleaning.py',
        execution_timeout=timedelta(minutes=10)
    )
    
    detect_bias = BashOperator(
        task_id='step5_detect_bias',
        bash_command=f'cd {PROJECT_DIR} && python src/data/step5_bias_detection_with_explicit_slicing.py',
        execution_timeout=timedelta(minutes=10)
    )
    
    detect_anomalies = BashOperator(
        task_id='step6_detect_anomalies',
        bash_command=f'cd {PROJECT_DIR} && python src/validation/step6_anomaly_detection.py',
        execution_timeout=timedelta(minutes=10)
    )
    
    end = BashOperator(
        task_id='end',
        bash_command='echo "Pipeline completed"'
    )
    
    # Pipeline flow - clean spacing
    start >> collect_data >> clean_data >> engineer_features >> merge_data >> clean_merged >> detect_bias >> detect_anomalies >> end
