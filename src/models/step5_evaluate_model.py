"""
Step 5: Evaluate Model
Comprehensive model evaluation with visualizations
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow import keras
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)

import mlflow


def load_model_and_data():
    """Load trained model and test data"""
    
    print("📥 Loading model and test data...")
    
    # Load model
    model = keras.models.load_model('models/lstm_baseline/model.h5')
    
    # Load scaler
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    # Load test data
    X_test = pd.read_csv('data/features/X_test.csv')
    y_test = pd.read_csv('data/features/y_test.csv').values.ravel()
    
    # Scale and reshape
    X_test_scaled = scaler.transform(X_test)
    X_test_lstm = X_test_scaled.reshape(X_test_scaled.shape[0], 12, -1)
    
    print(f"  ✓ Model and data loaded")
    
    return model, X_test_lstm, y_test


def generate_predictions(model, X_test):
    """Generate predictions"""
    
    print("\n🔮 Generating predictions...")
    
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).ravel()
    
    print(f"  ✓ Predictions generated")
    
    return y_pred, y_pred_proba


def calculate_metrics(y_test, y_pred, y_pred_proba):
    """Calculate all evaluation metrics"""
    
    print("\n📊 Calculating metrics...")
    
    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # ROC AUC
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Precision-Recall
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)
    
    metrics = {
        'accuracy': report['accuracy'],
        'precision': report['1']['precision'],
        'recall': report['1']['recall'],
        'f1_score': report['1']['f1-score'],
        'roc_auc': roc_auc,
        'avg_precision': avg_precision,
        'confusion_matrix': cm.tolist()
    }
    
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1-Score: {metrics['f1_score']:.4f}")
    print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
    
    return metrics, report, cm, (fpr, tpr), (precision, recall)


def plot_confusion_matrix(cm):
    """Plot confusion matrix"""
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('models/metrics/confusion_matrix.png', dpi=300)
    plt.close()
    
    print("  ✓ Confusion matrix saved")


def plot_roc_curve(fpr, tpr, roc_auc):
    """Plot ROC curve"""
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('models/metrics/roc_curve.png', dpi=300)
    plt.close()
    
    print("  ✓ ROC curve saved")


def evaluate_model():
    """Main evaluation function"""
    
    print("\n" + "="*70)
    print("STEP 5: MODEL EVALUATION")
    print("="*70)
    
    # Load model and data
    model, X_test, y_test = load_model_and_data()
    
    # Generate predictions
    y_pred, y_pred_proba = generate_predictions(model, X_test)
    
    # Calculate metrics
    metrics, report, cm, roc_data, pr_data = calculate_metrics(y_test, y_pred, y_pred_proba)
    
    # Generate plots
    print("\n📈 Generating visualizations...")
    Path('models/metrics').mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(cm)
    plot_roc_curve(roc_data[0], roc_data[1], metrics['roc_auc'])
    
    # Save metrics
    with open('models/metrics/evaluation_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✅ Evaluation complete!")
    print(f"  Metrics saved to: models/metrics/evaluation_metrics.json")
    
    # Log to MLflow
    with mlflow.start_run():
        mlflow.log_metrics(metrics)
        mlflow.log_artifact('models/metrics/confusion_matrix.png')
        mlflow.log_artifact('models/metrics/roc_curve.png')
    
    return metrics


if __name__ == "__main__":
    evaluate_model()