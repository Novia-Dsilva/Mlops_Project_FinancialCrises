# model_monitoring_dashboard.py
"""
Complete Monitoring Dashboard for Financial Stress Test Generator
Monitors: VAE, Predictive Models (5), and Anomaly Detection
Shows real-time drift status and performance metrics
"""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from google.cloud import storage
import json
from datetime import datetime

app = FastAPI(
    title="Financial Stress Test - Model Monitoring Dashboard",
    description="Real-time monitoring for VAE, Predictive, and Anomaly Detection models"
)

# Configuration
GCP_PROJECT = "ninth-iris-422916-f2"
GCP_BUCKET = "mlops-financial-stress-data"

@app.get("/")
async def root():
    """Root endpoint - API information"""
    return {
        "service": "Financial Stress Test - Model Monitoring",
        "version": "1.0",
        "status": "running",
        "models_monitored": {
            "vae": "Stress Scenario Generator",
            "predictive": "XGBoost/LightGBM (5 targets)",
            "anomaly": "Risk Scoring - Isolation Forest"
        },
        "endpoints": {
            "/": "API info",
            "/health": "Health check",
            "/report": "Monitoring dashboard (HTML)",
            "/api/drift-status": "Drift status (JSON)"
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "model-monitoring-dashboard"
    }

@app.get("/api/drift-status")
async def drift_status_api():
    """
    API endpoint returning drift status as JSON
    Useful for programmatic access
    """
    
    try:
        client = storage.Client(project=GCP_PROJECT)
        bucket = client.bucket(GCP_BUCKET)
        
        # Get latest drift report
        blobs = list(bucket.list_blobs(prefix='monitoring/drift_reports/drift_'))
        
        if blobs:
            latest_blob = sorted(blobs, key=lambda x: x.time_created, reverse=True)[0]
            drift_data = json.loads(latest_blob.download_as_text())
            
            return {
                "status": "success",
                "data": drift_data,
                "last_updated": drift_data.get('timestamp', 'Unknown')
            }
        else:
            return {
                "status": "no_data",
                "message": "No drift reports found yet",
                "last_updated": None
            }
    
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/report")
async def monitoring_dashboard():
    """
    Complete HTML Monitoring Dashboard
    Shows real-time status for all 3 model types
    """
    
    try:
        # Download latest drift report from GCS
        client = storage.Client(project=GCP_PROJECT)
        bucket = client.bucket(GCP_BUCKET)
        
        # Get all drift reports
        blobs = list(bucket.list_blobs(prefix='monitoring/drift_reports/drift_'))
        
        if blobs:
            # Get latest report
            latest_blob = sorted(blobs, key=lambda x: x.time_created, reverse=True)[0]
            drift_data = json.loads(latest_blob.download_as_text())
            
            # Get historical data for trend charts (last 30 checks)
            recent_blobs = sorted(blobs, key=lambda x: x.time_created, reverse=True)[:30]
            historical_data = []
            
            for blob in reversed(recent_blobs):
                try:
                    data = json.loads(blob.download_as_text())
                    historical_data.append({
                        'timestamp': data.get('timestamp', '')[:10],  # Date only
                        'vae_ks': data.get('models', {}).get('vae', {}).get('avg_ks', 0),
                        'pred_ks': data.get('models', {}).get('predictive', {}).get('avg_feature_ks', 0),
                        'anom_roc': data.get('models', {}).get('anomaly', {}).get('current_roc_auc', 0)
                    })
                except:
                    continue
        else:
            # No data yet - use defaults
            drift_data = {
                'timestamp': 'No data yet',
                'models': {
                    'vae': {'drift_detected': False, 'avg_ks': 0.0, 'pass_rate': 0.0},
                    'predictive': {'drift_detected': False, 'avg_feature_ks': 0.0, 'shift_rate': 0.0},
                    'anomaly': {'drift_detected': False, 'current_roc_auc': 0.0}
                },
                'any_drift': False
            }
            historical_data = []
    
    except Exception as e:
        print(f"Error loading drift data: {e}")
        drift_data = {
            'timestamp': f'Error: {e}',
            'models': {
                'vae': {'drift_detected': False, 'avg_ks': 0.0, 'pass_rate': 0.0},
                'predictive': {'drift_detected': False, 'avg_feature_ks': 0.0, 'shift_rate': 0.0},
                'anomaly': {'drift_detected': False, 'current_roc_auc': 0.0}
            },
            'any_drift': False
        }
        historical_data = []
    
    # Extract metrics for each model
    vae_data = drift_data.get('models', {}).get('vae', {})
    pred_data = drift_data.get('models', {}).get('predictive', {})
    anom_data = drift_data.get('models', {}).get('anomaly', {})
    
    # VAE metrics
    vae_ks = vae_data.get('avg_ks', 0.0)
    vae_drift = vae_data.get('drift_detected', False)
    vae_pass_rate = vae_data.get('pass_rate', 0.0)
    vae_failed = vae_data.get('failed_features', [])
    
    # Predictive metrics
    pred_ks = pred_data.get('avg_feature_ks', 0.0)
    pred_drift = pred_data.get('drift_detected', False)
    pred_shift_rate = pred_data.get('shift_rate', 0.0)
    pred_shifted = pred_data.get('shifted_features', [])
    
    # Anomaly metrics
    anom_roc = anom_data.get('current_roc_auc', 0.0)
    anom_drift = anom_data.get('drift_detected', False)
    anom_drop = anom_data.get('roc_drop_pct', 0.0)
    
    # Overall status
    any_drift = drift_data.get('any_drift', False)
    timestamp = drift_data.get('timestamp', 'Unknown')
    
    # Status colors
    vae_color = "#10b981" if vae_ks >= 0.70 else "#ef4444"
    vae_bg = "#d1fae5" if vae_ks >= 0.70 else "#fee2e2"
    
    pred_color = "#10b981" if pred_ks >= 0.70 else "#ef4444"
    pred_bg = "#d1fae5" if pred_ks >= 0.70 else "#fee2e2"
    
    anom_color = "#10b981" if anom_roc >= 0.75 else "#ef4444"
    anom_bg = "#d1fae5" if anom_roc >= 0.75 else "#fee2e2"
    
    # Prepare chart data
    if historical_data:
        chart_labels = str([d['timestamp'] for d in historical_data])
        chart_vae = str([d['vae_ks'] for d in historical_data])
        chart_pred = str([d['pred_ks'] for d in historical_data])
        chart_anom = str([d['anom_roc'] for d in historical_data])
    else:
        chart_labels = "[]"
        chart_vae = "[]"
        chart_pred = "[]"
        chart_anom = "[]"
    
    # Generate HTML
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Model Monitoring Dashboard - Financial Stress Test</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            
            body {{ 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }}
            
            .container {{ max-width: 1400px; margin: 0 auto; }}
            
            .header {{ 
                background: white;
                padding: 40px;
                border-radius: 20px;
                margin-bottom: 30px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            }}
            
            .header h1 {{ 
                color: #1a202c;
                font-size: 42px;
                font-weight: 800;
                margin-bottom: 12px;
                background: linear-gradient(135deg, #667eea, #764ba2);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }}
            
            .header p {{ 
                color: #718096; 
                font-size: 18px; 
                margin-bottom: 20px;
            }}
            
            .timestamp {{ 
                background: linear-gradient(135deg, #f7fafc, #edf2f7);
                padding: 12px 28px;
                border-radius: 30px;
                display: inline-block;
                font-size: 14px;
                font-weight: 600;
                color: #2d3748;
                border: 2px solid #e2e8f0;
            }}
            
            .alert-banner {{
                background: {'linear-gradient(135deg, #fee2e2, #fecaca)' if any_drift else 'linear-gradient(135deg, #d1fae5, #a7f3d0)'};
                border-left: 6px solid {'#ef4444' if any_drift else '#10b981'};
                padding: 30px;
                border-radius: 15px;
                margin-bottom: 30px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.15);
            }}
            
            .alert-banner h2 {{
                color: {'#991b1b' if any_drift else '#065f46'};
                font-size: 28px;
                font-weight: 700;
                margin-bottom: 12px;
            }}
            
            .alert-banner p {{
                color: {'#7f1d1d' if any_drift else '#047857'};
                font-size: 16px;
                line-height: 1.6;
            }}
            
            .grid {{ 
                display: grid; 
                grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); 
                gap: 25px;
                margin-bottom: 30px;
            }}
            
            .model-card {{ 
                background: white;
                padding: 40px;
                border-radius: 20px;
                box-shadow: 0 15px 50px rgba(0,0,0,0.2);
                transition: all 0.3s ease;
                border-top: 5px solid transparent;
            }}
            
            .model-card:hover {{ 
                transform: translateY(-8px);
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            }}
            
            .model-card.vae {{ border-top-color: #667eea; }}
            .model-card.predictive {{ border-top-color: #f093fb; }}
            .model-card.anomaly {{ border-top-color: #4facfe; }}
            
            .model-card h2 {{ 
                color: #1a202c;
                font-size: 26px;
                font-weight: 700;
                margin-bottom: 8px;
                display: flex;
                align-items: center;
            }}
            
            .model-card h2 .emoji {{ 
                font-size: 36px; 
                margin-right: 14px; 
            }}
            
            .model-subtitle {{
                color: #718096;
                font-size: 15px;
                margin-bottom: 25px;
                font-weight: 500;
            }}
            
            .metric-huge {{ 
                font-size: 72px; 
                font-weight: 800; 
                margin: 25px 0;
                line-height: 1;
                letter-spacing: -2px;
            }}
            
            .threshold-text {{
                color: #718096;
                font-size: 16px;
                margin: 18px 0;
                font-weight: 500;
            }}
            
            .status-badge {{ 
                display: inline-block;
                padding: 12px 28px;
                border-radius: 30px;
                font-weight: 700;
                font-size: 15px;
                margin-top: 18px;
                letter-spacing: 0.5px;
            }}
            
            .detail-box {{
                background: linear-gradient(135deg, #f7fafc, #edf2f7);
                padding: 25px;
                border-radius: 12px;
                margin-top: 25px;
                border-left: 4px solid #cbd5e0;
            }}
            
            .detail-box p {{
                color: #4a5568;
                margin: 10px 0;
                line-height: 1.8;
                font-size: 14px;
            }}
            
            .detail-box strong {{
                color: #2d3748;
                font-weight: 700;
            }}
            
            .chart-container {{
                background: white;
                padding: 40px;
                border-radius: 20px;
                box-shadow: 0 15px 50px rgba(0,0,0,0.2);
                margin-bottom: 30px;
            }}
            
            .chart-container h2 {{
                color: #1a202c;
                margin-bottom: 30px;
                font-size: 26px;
                font-weight: 700;
            }}
            
            .info-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
                gap: 20px;
                margin-top: 25px;
            }}
            
            .info-card {{
                background: linear-gradient(135deg, #f7fafc, #edf2f7);
                padding: 25px;
                border-radius: 12px;
                text-align: center;
                transition: transform 0.3s;
            }}
            
            .info-card:hover {{
                transform: translateY(-3px);
            }}
            
            .info-card h4 {{
                color: #4a5568;
                font-size: 12px;
                text-transform: uppercase;
                margin-bottom: 12px;
                letter-spacing: 1px;
                font-weight: 700;
            }}
            
            .info-card .value {{
                color: #1a202c;
                font-size: 28px;
                font-weight: 800;
            }}
            
            .info-card .subtitle {{
                color: #718096;
                font-size: 13px;
                margin-top: 8px;
            }}
            
            .footer {{
                text-align: center;
                color: white;
                margin-top: 50px;
                padding: 40px 20px;
            }}
            
            .footer p {{
                margin: 10px 0;
                text-shadow: 0 2px 4px rgba(0,0,0,0.2);
            }}
            
            @media (max-width: 768px) {{
                .grid {{ grid-template-columns: 1fr; }}
                .header h1 {{ font-size: 32px; }}
                .metric-huge {{ font-size: 56px; }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <!-- Header -->
            <div class="header">
                <h1>🏦 Financial Stress Test Generator</h1>
                <p>Real-time Model Performance Monitoring & Drift Detection System</p>
                <div class="timestamp">
                    📅 Last Checked: {timestamp[:19] if len(timestamp) > 19 else timestamp}
                </div>
            </div>
            
            <!-- Overall Status Alert -->
            <div class="alert-banner">
                <h2>{'🚨 Drift Detected - Retraining In Progress' if any_drift else '✅ All Systems Operational'}</h2>
                <p>
                    {
                        'One or more models have detected drift. Automated retraining has been triggered to maintain optimal prediction quality. You will be notified when retraining completes.' 
                        if any_drift else 
                        'All models are performing within acceptable thresholds. Continuous monitoring is active. No action required at this time.'
                    }
                </p>
            </div>
            
            <!-- Model Status Cards -->
            <div class="grid">
                <!-- VAE MODEL CARD -->
                <div class="model-card vae">
                    <h2><span class="emoji">🎯</span> VAE Model</h2>
                    <p class="model-subtitle">Stress Scenario Generator</p>
                    
                    <div class="metric-huge" style="color: {vae_color};">
                        {vae_ks:.3f}
                    </div>
                    
                    <p class="threshold-text">
                        <strong>Reconstruction KS Statistic</strong><br>
                        Threshold: ≥ 0.70
                    </p>
                    
                    <span class="status-badge" style="background: {vae_bg}; color: {vae_color};">
                        {('⚠️ Drift Detected - Retraining' if vae_drift else '✅ Healthy')}
                    </span>
                    
                    <div class="detail-box">
                        <p><strong>📊 Performance Metrics:</strong></p>
                        <p>• Reconstruction KS: {vae_ks:.4f}</p>
                        <p>• Feature Pass Rate: {vae_pass_rate*100:.1f}%</p>
                        <p>• Detection Method: Encode-Decode Test</p>
                        {f'<p style="color: #ef4444;">• Failed Features: {len(vae_failed)}</p>' if vae_failed else ''}
                        <p></p>
                        <p><strong>🔍 How It Works:</strong></p>
                        <p>Tests if VAE can reconstruct normal macroeconomic data. If reconstruction quality drops (KS < 0.70), the model has degraded and needs retraining.</p>
                        <p></p>
                        <p style="font-size: 11px; color: #718096; font-family: monospace;">
                        Production: gs://.../vae/deployment/best_model_deployment.pkl
                        </p>
                    </div>
                </div>
                
                <!-- PREDICTIVE MODELS CARD -->
                <div class="model-card predictive">
                    <h2><span class="emoji">📈</span> Predictive Models</h2>
                    <p class="model-subtitle">XGBoost / LightGBM (5 Targets)</p>
                    
                    <div class="metric-huge" style="color: {pred_color};">
                        {pred_ks:.3f}
                    </div>
                    
                    <p class="threshold-text">
                        <strong>Average Feature KS</strong><br>
                        Threshold: ≥ 0.70
                    </p>
                    
                    <span class="status-badge" style="background: {pred_bg}; color: {pred_color};">
                        {('⚠️ Drift Detected - Retraining' if pred_drift else '✅ Healthy')}
                    </span>
                    
                    <div class="detail-box">
                        <p><strong>📊 Distribution Metrics:</strong></p>
                        <p>• Feature Distribution KS: {pred_ks:.4f}</p>
                        <p>• Feature Shift Rate: {pred_shift_rate*100:.1f}%</p>
                        <p>• Detection Method: Input Distribution Test</p>
                        {f'<p style="color: #ef4444;">• Shifted Features: {len(pred_shifted)}</p>' if pred_shifted else ''}
                        <p></p>
                        <p><strong>🔍 How It Works:</strong></p>
                        <p>Compares input feature distributions between training and current data. If >30% features shifted or avg KS < 0.70, input data has changed significantly.</p>
                        <p></p>
                        <p><strong>🎯 Models:</strong> Revenue, EPS, Debt-Equity, Profit Margin, Stock Return</p>
                        <p style="font-size: 11px; color: #718096; font-family: monospace;">
                        Production: gs://.../models/best_models/[target]_best.pkl
                        </p>
                    </div>
                </div>
                
                <!-- ANOMALY DETECTION CARD -->
                <div class="model-card anomaly">
                    <h2><span class="emoji">⚠️</span> Anomaly Detection</h2>
                    <p class="model-subtitle">Risk Scoring - Isolation Forest</p>
                    
                    <div class="metric-huge" style="color: {anom_color};">
                        {anom_roc:.3f}
                    </div>
                    
                    <p class="threshold-text">
                        <strong>ROC-AUC Score</strong><br>
                        Threshold: ≥ 0.75
                    </p>
                    
                    <span class="status-badge" style="background: {anom_bg}; color: {anom_color};">
                        {('⚠️ Drift Detected - Retraining' if anom_drift else '✅ Healthy')}
                    </span>
                    
                    <div class="detail-box">
                        <p><strong>📊 Performance Metrics:</strong></p>
                        <p>• Current ROC-AUC: {anom_roc:.4f}</p>
                        <p>• Performance Drop: {anom_drop:.1f}%</p>
                        <p>• Detection Method: ROC-AUC Test</p>
                        <p></p>
                        <p><strong>🔍 How It Works:</strong></p>
                        <p>Re-evaluates model on validation set. If ROC-AUC falls below 0.75 or drops >5%, the model can no longer accurately identify at-risk companies.</p>
                        <p></p>
                        <p style="font-size: 11px; color: #718096; font-family: monospace;">
                        Production: gs://.../anomaly_detection/Isolation_Forest/model.pkl
                        </p>
                    </div>
                </div>
            </div>
            
            <!-- Performance Trends Chart -->
            <div class="chart-container">
                <h2>📈 Model Performance Trends (Last 30 Checks)</h2>
                <canvas id="trendsChart" style="max-height: 400px;"></canvas>
            </div>
            
            <!-- System Configuration -->
            <div class="chart-container">
                <h2>⚙️ Monitoring System Configuration</h2>
                
                <div class="info-grid">
                    <div class="info-card">
                        <h4>📅 Monitoring Schedule</h4>
                        <div class="value">Daily</div>
                        <p class="subtitle">2:00 AM UTC</p>
                    </div>
                    
                    <div class="info-card">
                        <h4>🔄 Auto-Retraining</h4>
                        <div class="value">Enabled</div>
                        <p class="subtitle">On drift detection</p>
                    </div>
                    
                    <div class="info-card">
                        <h4>📧 Notifications</h4>
                        <div class="value">Active</div>
                        <p class="subtitle">Email alerts</p>
                    </div>
                    
                    <div class="info-card">
                        <h4>💾 Model Storage</h4>
                        <div class="value">GCS</div>
                        <p class="subtitle">Cloud backup</p>
                    </div>
                    
                    <div class="info-card">
                        <h4>📊 Cloud Monitoring</h4>
                        <div class="value">Enabled</div>
                        <p class="subtitle">Google Cloud</p>
                    </div>
                    
                    <div class="info-card">
                        <h4>🎯 Total Models</h4>
                        <div class="value">7</div>
                        <p class="subtitle">VAE + 5 Predictive + Anomaly</p>
                    </div>
                </div>
            </div>
            
            <!-- Drift Detection Methodology -->
            <div class="chart-container">
                <h2>🔬 Drift Detection Methodology</h2>
                
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 25px; margin-top: 25px;">
                    <div style="background: linear-gradient(135deg, #ebf4ff, #dbeafe); padding: 30px; border-radius: 15px; border-left: 5px solid #667eea;">
                        <h3 style="color: #1e40af; margin-bottom: 15px; font-size: 20px; font-weight: 700;">🎯 VAE Model</h3>
                        <p style="color: #1e3a8a; line-height: 1.8; font-size: 14px;">
                            <strong>Test:</strong> Reconstruction Quality<br>
                            <strong>Process:</strong><br>
                            1. Sample 500 normal macro data points<br>
                            2. Encode → Decode (reconstruction)<br>
                            3. Compare original vs reconstructed<br>
                            4. Calculate KS statistic per feature<br>
                            <br>
                            <strong>Threshold:</strong> Avg KS ≥ 0.70<br>
                            <strong>Drift When:</strong> Can't reconstruct normal data accurately
                        </p>
                    </div>
                    
                    <div style="background: linear-gradient(135deg, #fef3f2, #fee2e2); padding: 30px; border-radius: 15px; border-left: 5px solid #f093fb;">
                        <h3 style="color: #991b1b; margin-bottom: 15px; font-size: 20px; font-weight: 700;">📈 Predictive Models</h3>
                        <p style="color: #7f1d1d; line-height: 1.8; font-size: 14px;">
                            <strong>Test:</strong> Input Feature Distribution<br>
                            <strong>Process:</strong><br>
                            1. Compare training vs current features<br>
                            2. Calculate KS per feature<br>
                            3. Identify shifted features<br>
                            4. Calculate shift rate<br>
                            <br>
                            <strong>Threshold:</strong> Avg KS ≥ 0.70 or <30% shifted<br>
                            <strong>Drift When:</strong> Input data distribution changed
                        </p>
                    </div>
                    
                    <div style="background: linear-gradient(135deg, #f0fdfa, #ccfbf1); padding: 30px; border-radius: 15px; border-left: 5px solid #4facfe;">
                        <h3 style="color: #065f46; margin-bottom: 15px; font-size: 20px; font-weight: 700;">⚠️ Anomaly Detection</h3>
                        <p style="color: #064e3b; line-height: 1.8; font-size: 14px;">
                            <strong>Test:</strong> ROC-AUC Performance<br>
                            <strong>Process:</strong><br>
                            1. Re-evaluate on validation set<br>
                            2. Calculate ROC-AUC<br>
                            3. Compare with baseline<br>
                            4. Check performance drop<br>
                            <br>
                            <strong>Threshold:</strong> ROC-AUC ≥ 0.75<br>
                            <strong>Drift When:</strong> Can't detect at-risk companies accurately
                        </p>
                    </div>
                </div>
            </div>
            
            <!-- Footer -->
            <div class="footer">
                <p style="font-size: 20px; font-weight: 700;">🏦 Financial Stress Test Generator</p>
                <p style="opacity: 0.9; margin-top: 10px; font-size: 16px;">MLOps Project | Northeastern University</p>
                <p style="opacity: 0.85; margin-top: 10px; font-size: 15px;">
                    Team: Novia Dsilva, Sanika Chaudhari, Parth Saraykar, Sushmitha Sudharsan, Priyanka Kumar, Sailee Choudhari
                </p>
                <p style="opacity: 0.75; margin-top: 15px; font-size: 14px;">
                    ⚡ Automated Monitoring • 🔍 Drift Detection • 🔄 Auto-Retraining • 📧 Email Alerts
                </p>
            </div>
        </div>
        
        <script>
            // Performance Trends Chart
            const ctx = document.getElementById('trendsChart').getContext('2d');
            
            const timestamps = {chart_labels};
            const vae_data = {chart_vae};
            const pred_data = {chart_pred};
            const anom_data = {chart_anom};
            
            const chart = new Chart(ctx, {{
                type: 'line',
                data: {{
                    labels: timestamps,
                    datasets: [
                        {{
                            label: 'VAE Reconstruction KS',
                            data: vae_data,
                            borderColor: '#667eea',
                            backgroundColor: 'rgba(102, 126, 234, 0.15)',
                            borderWidth: 3,
                            fill: true,
                            tension: 0.4,
                            pointRadius: 5,
                            pointHoverRadius: 8,
                            pointBackgroundColor: '#667eea',
                            pointBorderColor: '#fff',
                            pointBorderWidth: 2
                        }},
                        {{
                            label: 'Predictive Feature KS',
                            data: pred_data,
                            borderColor: '#f093fb',
                            backgroundColor: 'rgba(240, 147, 251, 0.15)',
                            borderWidth: 3,
                            fill: true,
                            tension: 0.4,
                            pointRadius: 5,
                            pointHoverRadius: 8,
                            pointBackgroundColor: '#f093fb',
                            pointBorderColor: '#fff',
                            pointBorderWidth: 2
                        }},
                        {{
                            label: 'Anomaly ROC-AUC',
                            data: anom_data,
                            borderColor: '#4facfe',
                            backgroundColor: 'rgba(79, 172, 254, 0.15)',
                            borderWidth: 3,
                            fill: true,
                            tension: 0.4,
                            pointRadius: 5,
                            pointHoverRadius: 8,
                            pointBackgroundColor: '#4facfe',
                            pointBorderColor: '#fff',
                            pointBorderWidth: 2
                        }},
                        {{
                            label: 'Threshold (0.70)',
                            data: Array(timestamps.length).fill(0.70),
                            borderColor: '#ef4444',
                            borderWidth: 3,
                            borderDash: [10, 5],
                            fill: false,
                            pointRadius: 0
                        }}
                    ]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: true,
                    plugins: {{
                        legend: {{
                            display: true,
                            position: 'top',
                            labels: {{
                                font: {{
                                    size: 14,
                                    weight: '600'
                                }},
                                padding: 20,
                                usePointStyle: true,
                                pointStyle: 'circle'
                            }}
                        }},
                        tooltip: {{
                            mode: 'index',
                            intersect: false,
                            backgroundColor: 'rgba(0, 0, 0, 0.85)',
                            titleFont: {{
                                size: 15,
                                weight: 'bold'
                            }},
                            bodyFont: {{
                                size: 14
                            }},
                            padding: 15,
                            cornerRadius: 10,
                            displayColors: true
                        }}
                    }},
                    scales: {{
                        y: {{
                            beginAtZero: false,
                            min: 0.5,
                            max: 1.0,
                            title: {{
                                display: true,
                                text: 'Performance Metric',
                                font: {{
                                    size: 15,
                                    weight: '700'
                                }},
                                color: '#1a202c'
                            }},
                            grid: {{
                                color: 'rgba(0, 0, 0, 0.08)',
                                lineWidth: 1
                            }},
                            ticks: {{
                                font: {{
                                    size: 13
                                }}
                            }}
                        }},
                        x: {{
                            title: {{
                                display: true,
                                text: 'Check Date',
                                font: {{
                                    size: 15,
                                    weight: '700'
                                }},
                                color: '#1a202c'
                            }},
                            grid: {{
                                display: false
                            }},
                            ticks: {{
                                font: {{
                                    size: 12
                                }},
                                maxRotation: 45,
                                minRotation: 45
                            }}
                        }}
                    }},
                    interaction: {{
                        mode: 'nearest',
                        axis: 'x',
                        intersect: false
                    }}
                }}
            }});
            
            // Auto-refresh every 5 minutes
            setTimeout(function() {{
                console.log('Auto-refreshing dashboard...');
                location.reload();
            }}, 300000);
        </script>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8080,
        log_level="info"
    )