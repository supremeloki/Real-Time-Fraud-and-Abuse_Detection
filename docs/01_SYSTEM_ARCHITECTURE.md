# 🏗️ Real-Time Fraud & Abuse Detection System: Architecture Overview

📋 This document outlines the high-level architecture of the Snapp Real-Time Fraud & Abuse Detection system, designed for high throughput, low latency, and robust model deployment.

## 🎯 1. Core Principles

*   **⚡ Real-Time Processing**: Leverage stream processing for immediate detection.
*   **📈 Scalability**: Built to handle Snapp's growing transaction volumes.
*   **🔧 Modularity**: Decoupled components for independent development and deployment.
*   **👁️ Observability**: Comprehensive monitoring and logging for operational health and model performance.
*   **🤖 Hybrid ML Approach**: Combining traditional ML (LightGBM) with advanced techniques (GNNs).

## 🏛️ 2. Architectural Components

🔧 The system is composed of several key components that interact to ingest data, generate features, run inference, and trigger actions.

### 📨 2.1 Data Ingestion Stream (Kafka)

*   **🔍 Source**: All Snapp operational events (ride requests, completions, payments, user actions, driver GPS updates) are published to Kafka topics.
*   **🎯 Role**: Provides a highly scalable, fault-tolerant backbone for real-time event delivery.
*   **🔑 Key Services**:
    *   `Kafka Event Producer`: Snapp backend services.
    *   `Kafka Event Consumer (src/ingestion_stream/kafka_event_consumer.py)`: Subscribes to relevant topics, validates schema, and forwards events for feature engineering.

### 💾 2.2 Real-Time Feature Store (Redis)

*   **🎯 Role**: Stores pre-computed, aggregated, and raw features for ultra-low latency retrieval during inference.
*   **📊 Data Types**: User/Driver historical activity, device telemetry, location context, transaction patterns.
*   **⚡ Technology**: Redis (for speed and in-memory operations).

### ⚙️ 2.3 Feature Engineering Layer (Spark Streaming / Flink / Python)

*   **🔄 Role**: Transforms raw event data into meaningful features for ML models.
*   **📋 Types**:
    *   **⚡ Real-Time Features (`src/feature_forge/real_time_features.py`)**: Computes features on the fly from incoming event streams (e.g., number of rides in last 5 mins, average speed).
    *   **📦 Batch Features (`src/feature_forge/batch_features.py`)**: Computes complex, aggregated features from historical data that don't require immediate updates (e.g., lifetime average fare, total promo usage).
    *   **🕸️ Graph Features (`src/feature_forge/graph_features.py`)**: Extracts features from graph data (e.g., node degrees, centrality measures, community detection) for GNN models.
*   **🛠️ Technology**: Python-based microservices for real-time, Spark/Flink for batch/streaming aggregates.

### 🚀 2.4 Model Prediction Engine (FastAPI)

*   **🎯 Role**: Hosts trained ML models, performs inference, and generates fraud scores.
*   **🔑 Key Services**:
    *   `Fraud Detection API (src/prediction_engine/fraud_detection_api.py)`: RESTful API endpoint for receiving requests and returning fraud predictions.
    *   `Inference Logic (src/prediction_engine/inference_logic.py)`: Orchestrates feature retrieval, model loading, and prediction execution.
    *   `Model Arsenal (src/model_arsenal/)`: Contains LightGBM and GNN models.
    *   `Latency Chamber (src/latency_chamber/)`: Optimized code for ultra-low latency inference.
*   **🌐 API Gateway**: Kubernetes Ingress with rate limiting (100 req/min), metrics endpoint (/metrics), and wildcard DNS support (*.snapp-fraud.local) for enhanced security and monitoring.

### 🧠 2.5 Decision & Action Layer

*   **⚖️ Role**: Interprets fraud scores and triggers appropriate actions.
*   **🧩 Components**:
    *   **📊 Thresholding Engine**: Applies configured thresholds (`conf/threshold_settings.yaml`) to fraud scores.
    *   **🚀 Action Dispatcher**:
        *   **🚫 Auto-Blocking**: For high-confidence fraud.
        *   **👥 Manual Review Queue**: For suspicious cases requiring human intervention (`src/feedback_loop/human_review_integration.py`).
        *   **⚠️ Warning/Notification**: For users/drivers exhibiting risky behavior.
        *   **🔔 Alerting**: To operational teams.
    *   **📝 Audit Trail**: Logs all decisions and actions for compliance and analysis.

### 🤖 2.6 MLOps & Monitoring

*   **📚 Model Registry (MLflow)**: Manages model versions, metadata, and artifacts.
*   **🧪 Experiment Lab (`experiment_lab/`)**:
    *   **⚖️ A/B Testing (`ab_testing/model_challenger_system.py`)**: For evaluating new models in production.
    *   **📊 Monitoring (`monitoring/`)**: Tracks model performance, data drift, and system health (Prometheus/Grafana).
*   **🔄 CI/CD Pipelines (`.github/workflows/`)**: Automates testing, building, and deploying models and services.

### 🔍 2.7 Interpretability Module (`src/interpretability_module/explanation_generator.py`)

*   **💡 Role**: Provides insights into why a specific decision was made, crucial for human review and model debugging.
*   **🛠️ Techniques**: SHAP, LIME, Feature Importance.

## 📊 3. Data Flow Diagram

```mermaid
graph TD
    A[Snapp Services] -->|Events (Ride, User, Payment)| B(Kafka Event Bus)
    B --> C{Kafka Event Consumer};
    C --> D[Real-Time Feature Store (Redis)]
    C --> E[Real-Time Feature Engineering]
    E --> D
    F[Historical Data Lake] --> G[Batch Feature Engineering]
    G --> D
    H[Graph Data Store] --> I[Graph Feature Engineering]
    I --> D
    D --> J[Prediction Engine API (FastAPI)]
    E --> J
    I --> J
    J --> K{Decision & Action Layer};
    K --> L[Auto-Block / Flag Account];
    K --> M[Human Review Queue];
    K --> N[Alerts & Notifications];
    J --> O[Model Monitoring];
    K --> O
    P[Data Scientist / Analyst] -- Feedback --> M
    M --> Q[Feedback Loop System];
    Q --> R[Model Retraining / Fine-tuning];
    R --> S[MLflow Model Registry];
    S --> J
    S --> T[CI/CD Pipelines];
    T --> J