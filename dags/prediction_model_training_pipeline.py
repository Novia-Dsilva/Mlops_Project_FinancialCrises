"""
Model Training Pipeline DAG - COMPLETE FLOW
Includes: Preprocessing → Training → Tuning → Validation → Bias → Selection

Flow:
1. Create targets
2. Drop features
3. Temporal split
4. Handle outliers
5. Train baseline models
6. Hyperparameter tuning
7. Model validation
8. Bias detection
9. Final selection
"""

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import sys

PROJECT_DIR = '/opt/airflow/project'
sys.path.insert(0, PROJECT_DIR)

# Import alerting if available
try:
    from src.monitoring.alerting import AlertManager
    ALERTING_AVAILABLE = True
except ImportError:
    print("WARNING: AlertManager not available")
    ALERTING_AVAILABLE = False

# ==============================================================================
# ALERTING CALLBACKS
# ==============================================================================

def task_failure_alert(context):
    """Alert on training task failure"""
    if not ALERTING_AVAILABLE:
        return
    try:
        task = context.get('task_instance')
        dag_run = context.get('dag_run')
        exception = context.get('exception')
        
        alert_manager = AlertManager()
        
        message = f"""
        Model Training Task Failed: {task.task_id}
        DAG: {task.dag_id}
        Execution Date: {dag_run.execution_date}
        Error: {str(exception) if exception else 'Check logs'}
        
        Log URL: {task.log_url}
        """
        
        alert_manager.send_alert(
            message=message,
            severity='ERROR',
            component=task.task_id,
            alert_type='MODEL_TRAINING_FAILURE'
        )
        print(f"Alert sent for {task.task_id} failure")
    except Exception as e:
        print(f"Failed to send alert: {str(e)}")

def training_success_alert(**context):
    """Alert on successful training completion"""
    if not ALERTING_AVAILABLE:
        return
    try:
        dag_run = context.get('dag_run')
        duration = dag_run.end_date - dag_run.start_date if dag_run.end_date else "N/A"
        
        alert_manager = AlertManager()
        
        message = f"""
        SUCCESS: Model Training Pipeline Complete!
        
        Execution Date: {dag_run.execution_date}
        Duration: {duration}
        
        Pipeline Summary:
        ✅ Data preprocessing completed
        ✅ Baseline models trained (XGBoost, LightGBM, LSTM)
        ✅ Hyperparameter tuning completed
        ✅ Model validation completed
        ✅ Crisis bias detection completed
        ✅ Final model selection completed
        
        Production models ready!
        Check: reports/final_selection/
        """
        
        alert_manager.send_alert(
            message=message,
            severity='INFO',
            component='model_training',
            alert_type='TRAINING_SUCCESS'
        )
        print("Training success alert sent")
    except Exception as e:
        print(f"Failed to send success alert: {str(e)}")

# ==============================================================================
# DAG CONFIGURATION
# ==============================================================================

default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'on_failure_callback': task_failure_alert,
}

# ==============================================================================
# DAG DEFINITION
# ==============================================================================

with DAG(
    'model_training_pipeline',
    default_args=default_args,
    description='Complete ML Pipeline: Preprocess → Train → Tune → Validate → Bias → Select',
    schedule_interval=None,  # Manual trigger
    start_date=days_ago(1),
    catchup=False,
    tags=['mlops', 'model-training', 'financial', 'ml'],
    max_active_runs=1,
) as dag:
    
    # ==========================================================================
    # PHASE 0: DATA PREPROCESSING (Before Training)
    # ==========================================================================
    
    create_targets = BashOperator(
        task_id='create_target_variables',
        bash_command=f"""
        cd {PROJECT_DIR} && \
        echo "📊 Creating target variables..." && \
        python src/models/create_target.py
        """,
        execution_timeout=timedelta(minutes=10),
    )
    
    drop_features = BashOperator(
        task_id='drop_unnecessary_features',
        bash_command=f"""
        cd {PROJECT_DIR} && \
        echo "✂️ Dropping unnecessary features..." && \
        python src/preprocessing/drop_features.py
        """,
        execution_timeout=timedelta(minutes=5),
    )
    
    temporal_split = BashOperator(
        task_id='temporal_train_test_split',
        bash_command=f"""
        cd {PROJECT_DIR} && \
        echo "📅 Creating temporal train/val/test split..." && \
        python src/preprocessing/temporal_split.py
        """,
        execution_timeout=timedelta(minutes=10),
    )
    
    handle_outliers = BashOperator(
        task_id='handle_outliers_after_split',
        bash_command=f"""
        cd {PROJECT_DIR} && \
        echo "🔍 Handling outliers after split..." && \
        python src/preprocessing/handle_outliers_after_split.py
        """,
        execution_timeout=timedelta(minutes=10),
    )
    
    # ==========================================================================
    # PHASE 1: BASELINE TRAINING (3 Models in Parallel)
    # ==========================================================================
    
    train_xgboost_baseline = BashOperator(
        task_id='train_xgboost_baseline',
        bash_command=f"""
        cd {PROJECT_DIR} && \
        echo "🚀 Training XGBoost baseline..." && \
        python src/models/xgboost_model.py --target all
        """,
        execution_timeout=timedelta(minutes=30),
    )
    
    train_lightgbm_baseline = BashOperator(
        task_id='train_lightgbm_baseline',
        bash_command=f"""
        cd {PROJECT_DIR} && \
        echo "🚀 Training LightGBM baseline..." && \
        python src/models/lightgbm_model.py --target all
        """,
        execution_timeout=timedelta(minutes=30),
    )
    
    train_lstm_baseline = BashOperator(
        task_id='train_lstm_baseline',
        bash_command=f"""
        cd {PROJECT_DIR} && \
        echo "🚀 Training LSTM baseline..." && \
        python src/models/lstm_model.py --target all
        """,
        execution_timeout=timedelta(minutes=30),
    )
    
    # ==========================================================================
    # PHASE 2: HYPERPARAMETER TUNING (3 Models in Parallel)
    # ==========================================================================
    
    tune_xgboost = BashOperator(
        task_id='tune_xgboost',
        bash_command=f"""
        cd {PROJECT_DIR} && \
        echo "⚙️ Tuning XGBoost hyperparameters..." && \
        python src/models/xgboost_hyperparameter_tuning.py --target all
        """,
        execution_timeout=timedelta(minutes=60),
        trigger_rule='none_failed',
    )
    
    tune_lightgbm = BashOperator(
        task_id='tune_lightgbm',
        bash_command=f"""
        cd {PROJECT_DIR} && \
        echo "⚙️ Tuning LightGBM hyperparameters..." && \
        python src/models/lightgbm_hyperparameter_tuning.py --target all
        """,
        execution_timeout=timedelta(minutes=60),
        trigger_rule='none_failed',
    )
    
    tune_lstm = BashOperator(
        task_id='tune_lstm',
        bash_command=f"""
        cd {PROJECT_DIR} && \
        echo "⚙️ Tuning LSTM hyperparameters..." && \
        python src/models/lstm_hyperparameter_tuning.py --target all
        """,
        execution_timeout=timedelta(minutes=60),
        trigger_rule='none_failed',
    )
    
    # ==========================================================================
    # PHASE 3: MODEL VALIDATION (Initial Selection Based on R²)
    # ==========================================================================
    
    model_validation = BashOperator(
        task_id='model_validation',
        bash_command=f"""
        cd {PROJECT_DIR} && \
        echo "✅ Validating models and selecting best performers..." && \
        python src/models/model_selection.py
        """,
        execution_timeout=timedelta(minutes=10),
    )
    
    # ==========================================================================
    # PHASE 4: BIAS DETECTION (Test ALL Models for Crisis Bias)
    # ==========================================================================
    
    bias_detection = BashOperator(
        task_id='bias_detection_all_models',
        bash_command=f"""
        cd {PROJECT_DIR} && \
        echo "🔍 Testing all models for crisis bias..." && \
        python src/evaluation/test_all_models_for_bias.py --target all
        """,
        execution_timeout=timedelta(minutes=30),
    )
    
    # ==========================================================================
    # PHASE 5: FINAL MODEL SELECTION (After Bias Analysis)
    # ==========================================================================
    
    final_selection = BashOperator(
        task_id='select_final_models',
        bash_command=f"""
        cd {PROJECT_DIR} && \
        echo "🎯 Making final model selection (R² + bias)..." && \
        python src/models/final_selection_after_bias_detection.py
        """,
        execution_timeout=timedelta(minutes=5),
    )
    
    # ==========================================================================
    # PHASE 6: GENERATE SUMMARY REPORT
    # ==========================================================================
    
    def print_training_summary(**context):
        """Print detailed summary of training results"""
        import json
        from pathlib import Path
        
        print("\n" + "="*80)
        print("🎉 MODEL TRAINING PIPELINE COMPLETE")
        print("="*80)
        
        report_file = Path('/opt/airflow/project/reports/final_selection/final_model_selection_after_bias.json')
        
        if report_file.exists():
            with open(report_file, 'r') as f:
                report = json.load(f)
            
            print("\n📊 FINAL PRODUCTION MODELS:")
            print("-" * 80)
            for target, info in report['final_selections'].items():
                bias_symbol = "✅" if info['bias_severity'] == 'NONE' else "⚠️" if info['bias_severity'] == 'MODERATE' else "🚨"
                switched = "⭐ SWITCHED" if info.get('switched_from_best_r2') else ""
                print(f"  {target:15} → {info['model']:20} (R²={info['test_r2']:.4f}, {bias_symbol} {info['bias_severity']}) {switched}")
            
            summary = report['summary']
            print(f"\n📈 SUMMARY:")
            print(f"  Total targets: {summary['total_targets']}")
            print(f"  Production ready: {summary['production_ready']}")
            print(f"  With warnings: {summary['with_warnings']}")
            print(f"  Rejected: {summary['rejected']}")
            
            print("\n📁 OUTPUTS:")
            print(f"  Models: /opt/airflow/project/models/")
            print(f"  Reports: /opt/airflow/project/reports/")
            print(f"  MLflow: /opt/airflow/project/mlruns/")
            
        else:
            print("\n⚠️ Final selection report not found!")
            print("Check logs for errors in previous tasks.")
        
        print("="*80 + "\n")
    
    training_summary = PythonOperator(
        task_id='print_training_summary',
        python_callable=print_training_summary,
    )
    
    # ==========================================================================
    # PHASE 7: SUCCESS NOTIFICATION
    # ==========================================================================
    
    training_success = PythonOperator(
        task_id='training_success_notification',
        python_callable=training_success_alert,
        trigger_rule='all_success',
    )
    
    # ==========================================================================
    # TASK DEPENDENCIES - COMPLETE FLOW
    # ==========================================================================
    
    # PHASE 0: Preprocessing (sequential)
    create_targets >> drop_features >> temporal_split >> handle_outliers
    
    # PHASE 1: Baseline training (parallel, after preprocessing)
    handle_outliers >> train_xgboost_baseline
    handle_outliers >> train_lightgbm_baseline
    handle_outliers >> train_lstm_baseline
    
    # PHASE 2: Tuning (parallel, after respective baseline)
    train_xgboost_baseline >> tune_xgboost
    train_lightgbm_baseline >> tune_lightgbm
    train_lstm_baseline >> tune_lstm
    
    tuning_models = [tune_xgboost, tune_lightgbm, tune_lstm]
    
    # PHASE 3: Model validation (after all tuning)
    tuning_models >> model_validation
    
    # PHASE 4: Bias detection (after validation)
    model_validation >> bias_detection
    
    # PHASE 5: Final selection (after bias)
    bias_detection >> final_selection
    
    # PHASE 6 & 7: Summary and notification
    final_selection >> training_summary >> training_success
