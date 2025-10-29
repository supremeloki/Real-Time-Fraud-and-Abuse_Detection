# 🚀 Deployment Operations Directory

This directory contains **all infrastructure-as-code configurations** for deploying the Real-Time Fraud & Abuse Detection system to various environments. ☁️

## 🗂️ Structure Overview

- **🐳 docker/** - Dockerfiles for containerizing different components
  - `Dockerfile.feature_processing` - Feature engineering container
  - `Dockerfile.inference_service` - Model inference API container
- **☸️ kubernetes/** - K8s manifests for microservices orchestration
  - API Gateway, Kafka, Redis, MLflow, Prometheus/Grafana
  - Feature processing, fraud detection, and model deployment services

## 🌍 Deployment Environments

Two primary deployment configurations exist:

### 🧪 Development Environment
- **📍 Location:** `../config/environments/dev.yaml`
- **⚡ Setup:** Lightweight with local services
- **🎯 Purpose:** Development and testing

### 🚀 Production Environment
- **📍 Location:** `../config/environments/prod.yaml`
- **📈 Scale:** Full Kubernetes deployment
- **🔒 Features:** High availability and monitoring

## ⚡ Quick Start

### 🐳 Local Docker Deployment
```bash
# Build and run feature processing service
docker build -f docker/Dockerfile.feature_processing -t fraud-feature-processing .
docker run -p 8080:8080 fraud-feature-processing

# Build and run inference service
docker build -f docker/Dockerfile.inference_service -t fraud-inference .
docker run -p 8081:8081 fraud-inference
```

### ☸️ Kubernetes Deployment
```bash
# Deploy to Kubernetes cluster
kubectl apply -f kubernetes/
kubectl get pods  # Check deployment status
```

## 📋 Prerequisites

- **🐳 Docker 20.10+**
- **☸️ Kubernetes 1.19+** (for K8s deployments)
- **⚓ Helm 3.x** (for chart deployments)
- **🛠️ kubectl** configured for target cluster

## ⚙️ Configuration

- **📁 Application config:** Use `../config/` for configuration
- **🔐 Secrets management:** `kubernetes/k8s_secrets_config.yaml`
- **🌍 Environment configs:** `../config/environments/`

## 📊 Monitoring & Observability

- **📈 Prometheus/Grafana:** `kubernetes/k8s_grafana_prometheus_setup.yaml`
- **📝 Centralized logging:** Integrated with application logging standards
- **📊 Metrics collection:** Configured in monitoring services

## 🔒 Security Notes

- **🔐 Production secrets:** Ensure proper encryption
- **🛡️ RBAC policies:** For Kubernetes deployments
- **🔍 Security scans:** Regular container image scanning recommended