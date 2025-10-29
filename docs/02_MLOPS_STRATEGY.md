# Real-Time Fraud & Abuse Detection System: MLOps Strategy

This document details the MLOps strategy for the Snapp Real-Time Fraud & Abuse Detection system, focusing on automation, reliability, and continuous improvement of ML models.

## 1. Goal

To establish a robust and automated workflow for developing, deploying, monitoring, and maintaining machine learning models in production, ensuring low-latency, high-precision fraud detection.

## 2. Key MLOps Pillars

### 2.1 Experiment Tracking & Reproducibility

*   **Tool**: MLflow Tracking (`mlflow_tracking_uri` in `conf/environments/`).
*   **Process**:
    *   All training runs (hyperparameters, metrics, artifacts, code versions) are logged.
    *   Models are automatically registered in the MLflow Model Registry upon successful training.
    *   Each experiment is tagged with dataset version, responsible team, and purpose.

### 2.2 Data Management

*   **Data Versioning**: DVC (Data Version Control) for tracking changes in datasets (training, validation, test) or explicit versioning within `data_vault/`.
*   **Data Validation**: Schemas (`data_vault/event_schemas/`) enforced during data ingestion and feature engineering to prevent data drift and ensure quality.
*   **Synthetic Data Generation**: `data_vault/fraud_pattern_simulator/generate_abuse_scenarios.py` and `data_vault/graph_topology_data/generate_collusion_graph.py` for creating diverse training scenarios and stress testing.

### 2.3 Feature Store

*   **Purpose**: Centralized management of features for consistency between training and inference, and real-time access.
*   **Tool**: Redis (`redis_host`, `redis_port` in `conf/environments/`) for real-time features.
*   **Process**:
    *   Real-time features computed by `src/feature_forge/real_time_features.py` are pushed to Redis.
    *   Batch features from `src/feature_forge/batch_features.py` are also pre-computed and stored.
    *   The prediction engine `src/prediction_engine/inference_logic.py` retrieves features directly from the Feature Store.

### 2.4 Model Training & Versioning

*   **Training Code**: Encapsulated in `src/model_arsenal/train_lightgbm.py` and `src/model_arsenal/train_gnn.py`.
*   **Model Registry**: MLflow Model Registry (`model_registry_metadata/fraud_model_versions.json` as a local representation, backed by MLflow server) is used to:
    *   Store different versions of LightGBM and GNN models.
    *   Track model lineage and metadata.
    *   Promote models from Staging to Production.

### 2.5 CI/CD for Models & Services

*   **Continuous Integration (`.github/workflows/ci_security_checks.yml`)**:
    *   Automated testing (unit, integration).
    *   Code quality checks (linting, formatting).
    *   Security scanning (dependency vulnerabilities, static analysis).
*   **Continuous Delivery/Deployment (`.github/workflows/cd_fraud_model_rollout.yml`)**:
    *   **Model Deployment**: Automated deployment of new model versions to production environments (triggered manually or via model promotion in MLflow).
    *   **Service Deployment**: Docker images (`deployment_ops/docker/`) built for inference services and deployed to Kubernetes (`deployment_ops/kubernetes/`).
    *   **Canary Deployments/Blue-Green**: Implemented via Kubernetes for gradual rollout and risk mitigation.

### 2.6 Model Monitoring & Alerting

*   **Tool**: Prometheus for metrics collection, Grafana for visualization.
*   **Metrics Tracked**:
    *   **Model Performance**: Precision, Recall, F1-score, AUC, False Positive Rate (FPR), False Negative Rate (FNR) on inferred data.
    *   **Data Drift**: Changes in input feature distributions.
    *   **Concept Drift**: Changes in the relationship between features and the target variable (fraud patterns evolve).
    *   **System Health**: Latency (`conf/threshold_settings.yaml#latency_budget_ms`), throughput, error rates of the prediction service.
*   **Alerting**: Automated alerts triggered for significant deviations in performance or data characteristics.

### 2.7 Feedback Loops & Retraining

*   **Human-in-the-Loop**: `src/feedback_loop/human_review_integration.py` captures manual review decisions and corrections.
*   **Automated Feedback**: Confirmed fraud/non-fraud labels are fed back into the training dataset.
*   **Retraining Policy**:
    *   **Scheduled Retraining**: Periodically (e.g., weekly/monthly, as specified in `conf/threshold_settings.yaml#auto_recalibration_frequency_days`) to adapt to evolving fraud patterns.
    *   **Triggered Retraining**: Based on detected data/concept drift or significant drops in model performance.
*   **A/B Testing (`experiment_lab/ab_testing/model_challenger_system.py`)**: New models are tested against the champion model in production to validate improvements before full rollout.

## 3. Ethical MLOps Considerations

*   **Transparency**: `src/interpretability_module/explanation_generator.py` provides model explanations for human reviewers.
*   **Fairness**: Regular bias audits on model predictions across different user/driver demographics.
*   **Privacy**: Adherence to data anonymization and privacy best practices (`docs/04_ETHICS_PRIVACY_GUIDE.md`).