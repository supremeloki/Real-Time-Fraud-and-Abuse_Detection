# 📁 Source Code Directory

This directory contains the **core source code** for the Real-Time Fraud & Abuse Detection system. 🏗️

## 🗂️ Structure Overview

- **🔍 correlation_engine/** - Cross-channel correlation analysis for fraud detection
- **💾 data_access/** - Data lakes, feature stores, and Redis cache connections
- **✅ data_quality/** - Data validation, drift detection, and quality checks
- **🧠 decision_engine/** - Fraud detection decisions and remediation orchestration
- **🧪 experiment_engine/** - A/B testing and experimental features management
- **🔍 explainability/** - SHAP explanations and model insights
- **⚙️ feature_forge/** - Batch and real-time feature engineering
- **🔄 feedback_loop/** - Model retraining and human review integration
- **🕸️ graph_processor/** - Graph anomaly detection and embeddings
- **📨 ingestion_stream/** - Kafka event consumption
- **💬 interpretability_module/** - Fraud prediction explanations
- **⚡ latency_chamber/** - Ultra-low latency optimizations
- **🎯 model_arsenal/** - GNN and LightGBM model implementations
- **📊 monitoring/** - Centralized alert monitoring and telemetry
- **🚀 prediction_engine/** - Fraud detection API and inference logic
- **👤 profile_builder/** - User behavioral profile construction
- **⚖️ risk_scoring/** - Dynamic risk policy engines
- **🔒 security/** - Threat intelligence feeds
- **🛠️ utils/** - Common utilities and configuration management

## 📦 Dependencies

- **🐍 Python 3.8+**
- **🔑 Key Libraries:** pandas, numpy, scikit-learn, lightgbm, tensorflow, kafka-python, redis

## 🚀 Running the Code

1. **📦 Install dependencies:** `pip install -r requirements.txt`
2. **⚙️ Set environment variables:** Copy `.env.template` to `.env` and configure
3. **▶️ Run specific modules:** Refer to individual module documentation

## 💡 Development Notes

- **📝 Follow logging standards:** `utils/logging_config.py`
- **🧪 Unit tests:** Located in corresponding `tests/` subdirectories
- **⚙️ Configuration:** Use `src/utils/config_manager.py` for config management