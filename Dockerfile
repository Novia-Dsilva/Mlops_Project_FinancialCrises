# FROM apache/airflow:2.7.1-python3.9

# USER airflow

# # Copy requirements file
# COPY requirements.txt /tmp/requirements.txt

# # Install Python dependencies
# RUN pip install --no-cache-dir --user -r /tmp/requirements.txt



FROM apache/airflow:2.7.1-python3.9

USER root

# Install system dependencies needed for model training
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# Copy requirements file
COPY requirements.txt /tmp/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --user -r /tmp/requirements.txt

# Create necessary directories for model outputs
RUN mkdir -p \
    /opt/airflow/models \
    /opt/airflow/reports/model_selection \
    /opt/airflow/reports/all_models_bias \
    /opt/airflow/reports/final_selection \
    /opt/airflow/mlruns

# Set environment variables for MLflow
ENV MLFLOW_TRACKING_URI=file:///opt/airflow/mlruns
ENV PYTHONPATH=/opt/airflow/project:$PYTHONPATH