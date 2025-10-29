# Financial Crisis Detection - MLOps Pipeline

	⁠*A production-ready MLOps pipeline for financial stress testing using dual-model architecture (VAE for scenario generation + XGBoost/LSTM for prediction)*

[![Coverage](https://img.shields.io/badge/coverage-84%25-brightgreen.svg)](htmlcov/index.html)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Table of Contents

•⁠  ⁠[Overview](#overview)
•⁠  ⁠[Architecture](#architecture)
•⁠  ⁠[Project Structure](#project-structure)
•⁠  ⁠[Prerequisites](#prerequisites)
•⁠  ⁠[Installation](#installation)
•⁠  ⁠[Configuration](#configuration)
•⁠  ⁠[Pipeline Execution](#pipeline-execution)
•⁠  ⁠[Data Validation](#data-validation)
•⁠  ⁠[Testing](#testing)
•⁠  ⁠[Monitoring & Alerts](#monitoring--alerts)
•⁠  ⁠[Reproducibility](#reproducibility)
•⁠  ⁠[Troubleshooting](#troubleshooting)

---

## Overview

This MLOps pipeline implements a comprehensive data processing workflow for financial stress testing with:

•⁠  ⁠*Point-in-time correctness* - 45-day reporting lag for quarterly financials

•⁠  ⁠*Dual-pipeline architecture* - VAE for scenarios + XGBoost/LSTM for predictions

•⁠  ⁠*Comprehensive validation* - 4 checkpoints with Great Expectations

•⁠  ⁠*Data versioning* - DVC for reproducibility

•⁠  ⁠*Quality assurance* - Anomaly detection, bias detection, drift detection

•⁠  ⁠*Production-ready* - Airflow orchestration, monitoring, alerting

•⁠  ⁠*Test coverage* - 84% (exceeds 75% requirement)

### *Data Sources:*
•⁠  ⁠*FRED* - 13 macroeconomic indicators (GDP, CPI, unemployment, etc.)
•⁠  ⁠*Yahoo Finance* - Market data (VIX, S&P 500) + 25 company stock prices
•⁠  ⁠*Alpha Vantage* - Company fundamentals (quarterly income statements & balance sheets)

### *Time Period:*
•⁠  ⁠*2005-01-01 to Present* (~20 years, covering 2008 crisis and 2020 COVID)

---

## Architecture

### *Pipeline Flow:*


Data Collection → Validate Raw → Clean → Validate Clean → 
Feature Engineering → Merge → Validate Merged → Clean Merged → 
Anomaly Detection → Bias Detection → Drift Detection → 
DVC Versioning → Testing → Ready for Modeling


### *Dual-Pipeline Design:*

*Pipeline 1 (VAE - Scenario Generation):*
•⁠  ⁠Input: ⁠ macro_features.csv ⁠ (FRED + Market)
•⁠  ⁠Purpose: Generate stress test scenarios
•⁠  ⁠Shape: ~5,500 rows × ~65 columns

*Pipeline 2 (XGBoost/LSTM - Prediction):*
•⁠  ⁠Input: ⁠ merged_features.csv ⁠ (FRED + Market + Company)
•⁠  ⁠Purpose: Predict company outcomes under scenarios
•⁠  ⁠Shape: ~188,000 rows × ~133 columns

---

## Project Structure

```text
Mlops_Project_FinancialCrises/
│
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── requirements-test.txt               # Testing dependencies
├── .env                                # Environment variables (DO NOT COMMIT)
├── .env.example                        # Template for .env
├── .gitignore                          # Git ignore rules
├── pytest.ini                          # Pytest configuration
├── dvc.yaml                            # DVC pipeline definition
├── params.yaml                         # Pipeline parameters
├── Makefile                            # Convenient commands
├── docker-compose.yml                  # Airflow setup
│
├── data/                               # Data directory (tracked by DVC)
│   ├── raw/                            # Raw data from APIs
│   │   ├── fred_raw.csv
│   │   ├── market_raw.csv
│   │   ├── company_prices_raw.csv
│   │   ├── company_income_raw.csv
│   │   └── company_balance_raw.csv
│   │
│   ├── clean/                          # Cleaned data (PIT correct)
│   │   ├── fred_clean.csv
│   │   ├── market_clean.csv
│   │   ├── company_prices_clean.csv
│   │   ├── company_balance_clean.csv
│   │   └── company_income_clean.csv
│   │
│   ├── features/                       # Feature-engineered data
│   │   ├── fred_features.csv
│   │   ├── market_features.csv
│   │   ├── company_features.csv
│   │   ├── macro_features.csv          # Pipeline 1
│   │   ├── merged_features.csv         # Pipeline 2
│   │   ├── macro_features_clean.csv
│   │   └── merged_features_clean.csv
│   │
│   ├── reports/                        # Cleaning & validation reports
│   ├── validation_reports/             # GE validation results
│   ├── anomaly_reports/                # Anomaly detection results
│   ├── bias_reports/                   # Bias detection results
│   └── drift_reports/                  # Drift detection results
│
├── src/                                # Source code
│   ├── validation/                     # Validation scripts
│   │   ├── robust_validator.py
│   │   ├── validation_utils.py
│   │   ├── ge_validator_base.py
│   │   ├── validate_checkpoint_1_raw.py
│   │   ├── validate_checkpoint_2_clean.py
│   │   ├── validate_checkpoint_3_merged.py
│   │   ├── validate_checkpoint_4_clean_merged.py
│   │   └── step5_anomaly_detection.py
│   │
│   ├── monitoring/                     # Monitoring & alerting
│   │   └── alerting.py
│   │
│   └── utils/                          # Utilities
│       ├── config.py                   # Configuration loader
│       └── dvc_helper.py               # DVC utilities
│
├── dags/                               # Airflow DAGs
│   └── financial_crisis_pipeline_dag.py
│
├── tests/                              # Test suite (84% coverage)
│   ├── conftest.py                     # Shared fixtures
│   ├── test_data_collection.py
│   ├── test_data_cleaning.py
│   ├── test_feature_engineering.py
│   ├── test_data_merging.py
│   ├── test_post_merge_cleaning.py
│   ├── test_anomaly_detection.py
│   ├── test_bias_detection.py
│   ├── test_drift_detection.py
│   ├── test_validation.py
│   ├── test_config.py
│   └── test_utils.py
│
├── logs/                               # Pipeline logs
│   └── pipeline_metrics.json
│
├── great_expectations/                 # GE configuration
│   └── great_expectations.yml
│
└── Pipeline Scripts (Root Directory)
    ├── step0_data_collection.py
    ├── step1_data_cleaning.py
    ├── step2_feature_engineering.py
    ├── step3_data_merging.py
    ├── step4_post_merge_cleaning.py
    ├── step5_bias_detection_with_explicit_slicing.py
    └── step6_anomaly_detection
    └── step7_drift_detection.py
```

---

## Prerequisites

### *Required Software:*
•⁠  ⁠Python 3.9+
•⁠  ⁠Git
•⁠  ⁠Docker & Docker Compose (for Airflow)
•⁠  ⁠pip (Python package manager)

### *Required Accounts (Free):*
•⁠  ⁠*Alpha Vantage API* - [Get free key](https://www.alphavantage.co/support/#api-key)
•⁠  ⁠*FRED API* (optional) - [Get free key](https://fred.stlouisfed.org/docs/api/api_key.html)
•⁠  ⁠*Slack Webhook* (optional) - [Setup guide](https://api.slack.com/messaging/webhooks)
•⁠  ⁠*Gmail App Password* (optional) - [Setup guide](https://support.google.com/accounts/answer/185833)

---

## Installation

### *Step 1: Clone Repository*

⁠ bash
git clone https://github.com/Novia-Dsilva/Mlops_Project_FinancialCrises.git
cd Mlops_Project_FinancialCrises
 ⁠

### *Step 2: Create Virtual Environment*

⁠ bash
## Create virtual environment
python3 -m venv fenv

## Activate (Mac/Linux)
source fenv/bin/activate

## Activate (Windows)
fenv\Scripts\activate
 ⁠

### *Step 3: Install Dependencies*

⁠ bash
# Install main dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install test dependencies
pip install -r requirements-test.txt
 ⁠

### *Step 4: Initialize DVC*

⁠ bash
## Initialize DVC
dvc init

## Add remote storage (choose one):

## Option A: Local remote (for testing)
dvc remote add -d local_remote /tmp/dvc-storage

## Option B: S3 remote (for production)
dvc remote add -d s3remote s3://your-bucket/dvc-storage
dvc remote modify s3remote access_key_id YOUR_AWS_KEY
dvc remote modify s3remote secret_access_key YOUR_AWS_SECRET

## Option C: Google Drive (free)
dvc remote add -d gdrive gdrive://YOUR_FOLDER_ID
 ⁠

### *Step 5: Setup Great Expectations*

⁠ bash
## Initialize Great Expectations (will be done automatically by validation scripts)
## Or manually:
great-expectations init
 ⁠

---

## Configuration

### *Create .env File*

Copy the template and fill in your values:

⁠ bash
## Copy template
cp .env.example .env

## Edit with your values
nano .env  # or use your favorite editor
 ⁠

### *Minimal .env Configuration:*

⁠ bash
### ===============================================================
## REQUIRED CONFIGURATION
### ===============================================================

## Airflow
AIRFLOW_UID=50000
AIRFLOW__CORE__FERNET_KEY=YOUR_FERNET_KEY_HERE
AIRFLOW__WEBSERVER__SECRET_KEY=YOUR_SECRET_KEY_HERE

## API Keys
ALPHA_VANTAGE_API_KEY=YOUR_ALPHA_VANTAGE_KEY_HERE

## DVC
DVC_REMOTE_TYPE=local
DVC_LOCAL_REMOTE=/tmp/dvc-storage

## Pipeline Parameters
START_DATE=2005-01-01
END_DATE=today
REPORTING_LAG_DAYS=45

### =================================================================
## OPTIONAL CONFIGURATION
### =================================================================

## Alerts (can disable)
SLACK_ALERTS_ENABLED=false
EMAIL_ALERTS_ENABLED=false

## Thresholds
ANOMALY_IQR_THRESHOLD=3.0
BIAS_REPRESENTATION_THRESHOLD=0.3
MAX_MISSING_PCT_CLEAN=5
 ⁠

### *Generate Required Keys:*

⁠ bash
## Generate Fernet key for Airflow
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

## Generate secret key for Airflow
python -c "import secrets; print(secrets.token_urlsafe(32))"
 ⁠

---

## Pipeline Execution

### *Option 1: Run Complete Pipeline (Recommended)*

⁠ bash
## Make run script executable
chmod +x run_pipeline.sh

## Run complete pipeline
./run_pipeline.sh
 ⁠

### *Option 2: Run Individual Steps*

⁠ bash
## Step 0: Data Collection (~18 minutes)
python step0_data_collection.py

## Checkpoint 1: Validate Raw Data
python src/validation/validate_checkpoint_1_raw.py

## Step 1: Data Cleaning
python step1_data_cleaning.py

## Checkpoint 2: Validate Clean Data
python src/validation/validate_checkpoint_2_clean.py

## Step 2: Feature Engineering
python step2_feature_engineering.py

## Step 3: Data Merging
python step3_data_merging.py

## Checkpoint 3: Validate Merged Data
python src/validation/validate_checkpoint_3_merged.py

## Step 3c: Clean Merged Data
python step3c_post_merge_cleaning.py

## Step 5: Anomaly Detection
python src/validation/step5_anomaly_detection.py

## Step 4: Bias Detection
python step4_bias_detection.py

## Step 6: Drift Detection
python step6_drift_detection.py

# DVC: Version all data
dvc add data/raw data/clean data/features
dvc push
git add data/*.dvc .dvc/config
git commit -m "Pipeline run complete"
git push
 ⁠

### *Option 3: Run with Airflow (Production)*

⁠ bash
## Step 1: Start Airflow
docker-compose up -d

## Step 2: Wait for initialization (~2 minutes)
docker-compose logs -f airflow-init

## Step 3: Access Airflow UI
### Open: http://localhost:8080
### Username: admin
### Password: admin

## Step 4: Trigger DAG from UI or CLI
docker-compose exec airflow-webserver airflow dags trigger financial_crisis_detection_pipeline

## Step 5: Monitor progress in UI
## View logs, task status, and execution graph
 ⁠

### *Option 4: Run with DVC Pipeline*

⁠ bash
## Run entire DVC pipeline
dvc repro

## Run specific stage
dvc repro data_cleaning

## Check pipeline status
dvc status
 ⁠
---

## Data Validation

### *Validation Checkpoints:*

The pipeline includes 4 validation checkpoints using Great Expectations:

*Checkpoint 1: Raw Data Validation*
•⁠  ⁠Files exist
•⁠  ⁠Required columns present
•⁠  ⁠Data types correct
•⁠  ⁠Reasonable row counts
•⁠  ⁠Date ranges valid

*Checkpoint 2: Clean Data Validation*
•⁠  ⁠Missing values handled (<5%)
•⁠  ⁠No inf values
•⁠  ⁠Duplicates removed
•⁠  ⁠Point-in-time correctness maintained
•⁠  ⁠Forward-fill only (no look-ahead bias)

*Checkpoint 3: Merged Data Validation*
•⁠  ⁠Merge quality (no data loss)
•⁠  ⁠Date alignment correct
•⁠  ⁠All companies have macro context
•⁠  ⁠No duplicate (Date, Company) pairs

*Checkpoint 4: Clean Merged Data Validation*
•⁠  ⁠Zero inf values (CRITICAL)
•⁠  ⁠Minimal missing (<2%)
•⁠  ⁠No duplicate columns with suffixes
•⁠  ⁠Valid financial ratios
•⁠  ⁠Proper data types

### *View Validation Results:*

⁠ bash
# View validation reports
ls -lh data/validation_reports/

# View specific report
cat data/validation_reports/ge_merged_features_clean_*.json | python -m json.tool

# View Great Expectations data docs
great-expectations docs build
# Then open: great_expectations/uncommitted/data_docs/local_site/index.html
 ⁠

---

## Testing

### *Run Tests:*

⁠ bash
## Run all tests with coverage
pytest --cov=src --cov=. --cov-report=html --cov-report=term-missing

## Or use Makefile
make test          # Run all tests
make coverage      # Run with coverage report

## Or use helper script
./run_tests.sh
 ⁠

### *Test Coverage:*

Current coverage: *84%* (exceeds 75% requirement)

⁠ bash
## View coverage report
open htmlcov/index.html  # Mac
xdg-open htmlcov/index.html  # Linux
 ⁠

### *Run Specific Tests:*

⁠ bash
## Test specific module
pytest tests/test_data_cleaning.py -v

## Test specific class
pytest tests/test_drift_detection.py::TestKSTestDriftDetection -v

## Run fast tests only
pytest -m "not slow"

## Run in parallel
pytest -n auto
 ⁠

---

## Monitoring & Alerts

### *Anomaly Detection:*

Detects three types of anomalies:
1.⁠ ⁠*Statistical outliers* - IQR method (>3 std deviations)
2.⁠ ⁠*Business rule violations* - Domain-specific rules (e.g., negative VIX)
3.⁠ ⁠*Temporal anomalies* - Sudden jumps (>50% change)

*Output:*
•⁠  ⁠Flag columns added (no data modification)
•⁠  ⁠Detailed reports in ⁠ data/anomaly_reports/ ⁠
•⁠  ⁠Alerts sent if critical anomalies found

### *Bias Detection:*

Performs data slicing across 3 dimensions:
1.⁠ ⁠*Company-level* - 25 companies
2.⁠ ⁠*Sector-level* - 9 sectors
3.⁠ ⁠*Temporal* - 5 time periods (pre-crisis, crisis, recovery, COVID, recent)

*Output:*
•⁠  ⁠Slice statistics in ⁠ data/bias_reports/ ⁠
•⁠  ⁠Mitigation recommendations
•⁠  ⁠Representation bias analysis

### *Drift Detection:*

Compares historical distributions:
•⁠  ⁠*Reference period:* 2005-2010
•⁠  ⁠*Current period:* 2020-2025
•⁠  ⁠*Method:* Kolmogorov-Smirnov test

*Output:*
•⁠  ⁠Drifted features report in ⁠ data/drift_reports/ ⁠
•⁠  ⁠Feature stability analysis
•⁠  ⁠Mitigation recommendations

### *Alert Configuration:*

Alerts are sent when:
•⁠  ⁠Validation checkpoint fails
•⁠  ⁠Critical anomalies detected (>10 critical issues)
•⁠  ⁠High drift detected (>20 features with p<0.01)
•⁠  ⁠Pipeline step exceeds time threshold (>1 hour)

*Enable Alerts in .env:*

⁠ bash
## Slack
SLACK_ALERTS_ENABLED=true
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL

## Email
EMAIL_ALERTS_ENABLED=true
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SENDER_EMAIL=your-email@gmail.com
SENDER_APP_PASSWORD=your_16_char_app_password
RECIPIENT_EMAILS=team@example.com,member@example.com
 ⁠
---

## Reproducibility

### *Pull Data from DVC:*

⁠ bash
## Pull all versioned data
dvc pull

## Pull specific stage
dvc pull data/raw.dvc
dvc pull data/clean.dvc
 ⁠

### *Recreate Entire Pipeline:*

⁠ bash
## Step 1: Clone repository
git clone https://github.com/yourusername/Mlops_Project_FinancialCrises.git
cd Mlops_Project_FinancialCrises

## Step 2: Setup environment
python3 -m venv fenv
source fenv/bin/activate
pip install -r requirements.txt

## Step 3: Configure
cp .env.example .env
## Edit .env with your API keys

## Step 4: Initialize DVC
dvc init
dvc remote add -d local_remote /tmp/dvc-storage

## Step 5: Pull data (if already versioned)
dvc pull

## Step 6: Or regenerate from scratch
python step0_data_collection.py
### ... run all steps

## Step 7: Run tests
pytest --cov=src --cov-report=html

## Step 8: Everything should match original results!
 ⁠

### *Version Control Workflow:*

⁠ bash
## After running pipeline successfully:

## 1. Version data with DVC
dvc add data/raw data/clean data/features
dvc push

## 2. Commit code and DVC files to Git
git add .
git commit -m "Pipeline run: $(date +%Y-%m-%d)"
git push

## 3. Tag release
git tag -a v1.0 -m "Validated pipeline with 84% test coverage"
git push --tags
 ⁠
---

## Key Features

### *1. Point-in-Time Correctness *

Ensures no look-ahead bias:

⁠ python
# Quarterly financials shifted +45 days
# Q1 2020 (Mar 31) → Available May 15 (45 days later)

# Forward fill ONLY (no backward fill)
# Missing values filled from past, never from future
 ⁠

*Why it matters:* Prevents using future information that wouldn't be available at prediction time.

### *2. Data Quality Assurance *

*4 Validation Checkpoints:*
•⁠  ⁠Raw data validation (schema, ranges, completeness)
•⁠  ⁠Clean data validation (missing <5%, no inf, no duplicates)
•⁠  ⁠Merged data validation (alignment, no data loss)
•⁠  ⁠Final validation (production-ready quality)

*Quality Metrics:*
•⁠  ⁠Zero inf values after cleaning
•⁠  ⁠<2% missing values in merged data
•⁠  ⁠No duplicate columns from merge operations
•⁠  ⁠Valid financial ratios

### *3. Bias Detection & Mitigation *

*Data Slicing Analysis:*
•⁠  ⁠25 company slices
•⁠  ⁠9 sector slices
•⁠  ⁠5 temporal slices

*Biases Detected:*
•⁠  ⁠Representation bias (sample imbalance)
•⁠  ⁠Distribution bias (feature behavior across groups)
•⁠  ⁠Temporal bias (data quality over time)

*Mitigation Applied:*
•⁠  ⁠Stratified train/test split by Sector
•⁠  ⁠Weighted loss function (weight ∝ 1/company_samples)
•⁠  ⁠Crisis periods included in validation set

### *4. Anomaly Detection *

*Detection Methods:*
•⁠  ⁠IQR outliers (statistical)
•⁠  ⁠Business rule violations (domain-specific)
•⁠  ⁠Temporal jumps (time-series)

*Crisis Awareness:*
•⁠  ⁠Distinguishes 2008-2009 and 2020 outliers (valid) from data errors
•⁠  ⁠Flags but doesn't remove crisis data

*Output:*
•⁠  ⁠16 flag columns added (e.g., ⁠ Stock_Return_1D_Outlier_Flag ⁠)
•⁠  ⁠Original data unchanged
•⁠  ⁠Detailed reports with severity levels

### *5. Comprehensive Testing *

*Test Coverage: 84%*

⁠ bash
## Coverage breakdown:
src/validation/step5_anomaly_detection.py  86%
step1_data_cleaning.py                     83%
step3c_post_merge_cleaning.py              85%
step6_drift_detection.py                   85%
step4_bias_detection.py                    85%
 ⁠

*Test Suite:*
•⁠  ⁠118 passing tests
•⁠  ⁠Unit tests (fixtures, mocking)
•⁠  ⁠Integration tests (end-to-end)
•⁠  ⁠Edge case testing

---

## Output Files

### *Raw Data (data/raw/):*
•⁠  ⁠⁠ fred_raw.csv ⁠ - 5,571 rows × 13 columns
•⁠  ⁠⁠ market_raw.csv ⁠ - 5,238 rows × 2 columns
•⁠  ⁠⁠ company_prices_raw.csv ⁠ - 129,569 rows (25 companies)
•⁠  ⁠⁠ company_income_raw.csv ⁠ - 2,016 quarters
•⁠  ⁠⁠ company_balance_raw.csv ⁠ - 2,016 quarters

### *Clean Data (data/clean/):*
•⁠  ⁠Same files with PIT correction applied
•⁠  ⁠Missing values handled
•⁠  ⁠Duplicates removed
•⁠  ⁠Outliers flagged (not removed)

### *Features (data/features/):*
•⁠  ⁠⁠ macro_features_clean.csv ⁠ - Pipeline 1 (5,571 × 67)
•⁠  ⁠⁠ merged_features_clean.csv ⁠ - Pipeline 2 (188,670 × 133)
•⁠  ⁠⁠ merged_features_clean_with_anomaly_flags.csv ⁠ - With anomaly flags (188,670 × 149)

### *Reports:*
•⁠  ⁠Validation reports (JSON)
•⁠  ⁠Anomaly reports (JSON + CSV)
•⁠  ⁠Bias reports (JSON + CSV)
•⁠  ⁠Drift reports (JSON + CSV)
•⁠  ⁠Test coverage report (HTML)

---

## Troubleshooting

### *Common Issues:*

*Issue 1: "ModuleNotFoundError: No module named 'src'"*

⁠ bash
## Solution: Add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:${PWD}"

## Or install as package
pip install -e .
 ⁠

*Issue 2: "Alpha Vantage API limit reached"*

⁠ bash
## Solution: Wait 60 seconds or use multiple keys
## Add to .env:
ALPHA_VANTAGE_API_KEY_BACKUP=SECOND_KEY_HERE
 ⁠

*Issue 3: "DVC push failed"*

⁠ bash
## Check remote configuration
dvc remote list

## Verify credentials
dvc remote modify local_remote --local access_key_id YOUR_KEY
 ⁠

*Issue 4: "pytest: error: unrecognized arguments: --cov"*

⁠ bash
## Install pytest-cov
pip install pytest-cov

## Or use alternative
coverage run -m pytest tests/
coverage html
 ⁠

*Issue 5: "Great Expectations validation failed"*

⁠ bash
## View detailed report
great-expectations docs build
# Open: great_expectations/uncommitted/data_docs/local_site/index.html

## Check which expectations failed
cat data/validation_reports/ge_*.json | python -m json.tool
 ⁠

---

## Documentation

### *Additional Resources:*

•⁠  ⁠[Great Expectations Docs](https://docs.greatexpectations.io/)
•⁠  ⁠[DVC Documentation](https://dvc.org/doc)
•⁠  ⁠[Airflow Documentation](https://airflow.apache.org/docs/)
•⁠  ⁠[Pytest Documentation](https://docs.pytest.org/)

### *Project Documentation:*

⁠ bash
## Generate API documentation
pip install pdoc3
pdoc --html --output-dir docs src/

## View docs
open docs/src/index.html
 ⁠
---

## Contributing

### *Development Workflow:*

⁠ bash
## 1. Create feature branch
git checkout -b feature/your-feature

## 2. Make changes
## Edit code...

## 3. Run tests
pytest

## 4. Run pipeline
python step1_data_cleaning.py  # etc.

## 5. Commit changes
git add .
git commit -m "Add: your feature"

## 6. Push and create PR
git push origin feature/your-feature
 ⁠

### *Code Quality:*

⁠ bash
## Run linting
pylint src/ step*.py

## Format code
black src/ step*.py

## Type checking
mypy src/
 ⁠

---

## Pipeline Statistics

### *Data Volume:*

| Stage | Files | Total Rows | Total Size |
|-------|-------|-----------|-----------|
| Raw | 5 | 142,437 | 18.22 MB |
| Clean | 5 | 142,437 | 26.45 MB |
| Features | 3 | 199,699 | 175.89 MB |
| Final | 2 | 194,241 | 151.01 MB |

### *Execution Time:*

| Step | Duration | Notes |
|------|----------|-------|
| Data Collection | ~18 min | Alpha Vantage rate limits |
| Data Cleaning | ~2 min | All datasets |
| Feature Engineering | ~3 min | Including quarterly→daily |
| Data Merging | ~1 min | Both pipelines |
| Validation (all) | ~5 min | 4 checkpoints |
| Anomaly Detection | ~2 min | Flag creation |
| *Total* | *~31 min* | End-to-end |

---

## Learning Outcomes

This pipeline demonstrates:

*MLOps Best Practices*
•⁠  ⁠Data versioning (DVC)
•⁠  ⁠Pipeline orchestration (Airflow)
•⁠  ⁠Automated validation (Great Expectations)
•⁠  ⁠Comprehensive testing (pytest, 84% coverage)
•⁠  ⁠Monitoring & alerting (Email, Slack)

*Data Engineering*
•⁠  ⁠Point-in-time correctness
•⁠  ⁠Quarterly to daily conversion
•⁠  ⁠Feature engineering (45+ features per dataset)
•⁠  ⁠Multi-source data merging

*Data Quality*
•⁠  ⁠Anomaly detection (flag-only, crisis-aware)
•⁠  ⁠Bias detection (data slicing across 3 dimensions)
•⁠  ⁠Drift detection (historical comparison)
•⁠  ⁠Validation at each stage

*Production Readiness*
•⁠  ⁠Error handling & retries
•⁠  ⁠Detailed logging
•⁠  ⁠Alert system
•⁠  ⁠Reproducibility (DVC + Docker)
•⁠  ⁠Documentation

---

## Support

For questions or issues:

1.⁠ ⁠Check [Troubleshooting](#troubleshooting) section
2.⁠ ⁠Review validation reports in ⁠ data/validation_reports/ ⁠
3.⁠ ⁠Check logs in ⁠ logs/ ⁠ directory
4.⁠ ⁠Open an issue on GitHub

---

## License

MIT License - see [LICENSE](LICENSE) file for details

---

## Acknowledgments

•⁠  ⁠*FRED* - Federal Reserve Economic Data
•⁠  ⁠*Yahoo Finance* - Market & stock price data
•⁠  ⁠*Alpha Vantage* - Company fundamentals
•⁠  ⁠*Great Expectations* - Data validation framework
•⁠  ⁠*DVC* - Data version control

---

## Quick Start Summary

⁠ bash
## 1. Clone and setup
git clone <repo-url>
cd Mlops_Project_FinancialCrises
python3 -m venv fenv && source fenv/bin/activate
pip install -r requirements.txt

## 2. Configure
cp .env.example .env
### Add your ALPHA_VANTAGE_API_KEY to .env

## 3. Initialize DVC
dvc init
dvc remote add -d local_remote /tmp/dvc-storage

## 4. Run pipeline
python step0_data_collection.py  # ~18 min
python src/validation/validate_checkpoint_1_raw.py
python step1_data_cleaning.py
python src/validation/validate_checkpoint_2_clean.py
python step2_feature_engineering.py
python step3_data_merging.py
python src/validation/validate_checkpoint_3_merged.py
python step3c_post_merge_cleaning.py
python src/validation/step5_anomaly_detection.py
python step4_bias_detection.py
python step6_drift_detection.py

## 5. Version data
dvc add data/raw data/clean data/features
dvc push

## 6. Run tests
pytest --cov=src --cov-report=html

# 7. Success! 
### - Data in: data/features/merged_features_clean.csv
### - Coverage: 84%
### - All validations passed


---

*Built by Mlops Group11 Team*

