# Real-Time Fraud & Abuse Detection System: SnappTech Demo Guide

This guide provides instructions for setting up and demonstrating the Snapp Real-Time Fraud & Abuse Detection system. It's designed to showcase the core functionalities, from data generation to real-time detection and model interpretability.

## 1. Prerequisites

*   Python 3.9+
*   Docker & Docker Compose
*   Kafka (running locally or accessible)
*   Redis (running locally or accessible)
*   MLflow Server (running locally or accessible)
*   Git

## 2. Setup Local Environment

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/SnappTech/real-time-fraud-detection.git
    cd real-time-fraud-detection
    ```

2.  **Create Virtual Environment & Install Dependencies**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3.  **Start Infrastructure (Kafka, Redis, MLflow)**:
    For a simplified local setup, you can use Docker Compose (assuming a `docker-compose.yml` for infrastructure components exists, though not part of the provided tree):
    ```bash
    # (Assuming a docker-compose.yml for Kafka, Redis, MLflow)
    docker-compose up -d
    ```
    Alternatively, ensure these services are running and accessible as configured in `conf/environments/dev.yaml`.

## 3. Demo Steps

### Step 3.1: Simulate Fraud Data

Generate synthetic ride events, including various fraud patterns (promo abuse, fake rides, collusion scenarios).

```bash
python data_vault/fraud_pattern_simulator/generate_abuse_scenarios.py
python data_vault/graph_topology_data/generate_collusion_graph.py
```
This will create `synthetic_fraud_events.csv`, `graph_nodes.csv`, and `graph_edges.csv`.

### Step 3.2: Train Initial LightGBM Model

Train the initial fraud detection model. The model will be logged to MLflow.

```bash
python src/model_arsenal/train_lightgbm.py --env dev
```
(Optional: Navigate to your MLflow UI at `http://localhost:5000` to see the experiment run and registered model.)

### Step 3.3: Launch Real-Time Feature Engineering & Inference Service

1.  **Start Kafka Consumer & Feature Processor**: This script will listen to a Kafka topic, process events, and update the Redis Feature Store.
    ```bash
    python src/ingestion_stream/kafka_event_consumer.py --env dev
    ```
2.  **Start Prediction Engine API**: This API will serve fraud predictions using the trained model.
    ```bash
    uvicorn src.prediction_engine.fraud_detection_api:app --host 0.0.0.0 --port 8000
    ```
    **Note**: The API is exposed via Kubernetes Ingress at `api.snapp-fraud.local` with rate limiting (100 req/min) and includes a `/metrics` endpoint for monitoring.

### Step 3.4: Stream Events and Get Real-Time Predictions

Use a separate script or the `snapptech_fraud_demo.ipynb` notebook to simulate streaming events to Kafka and querying the prediction API.

**Example `stream_and_query.py` (Hypothetical)**:
```python
# scripts/stream_and_query.py (Not part of main tree, for demo purposes)
import json
import time
import requests
from confluent_kafka import Producer

# Kafka Producer setup (replace with your broker)
producer_conf = {'bootstrap.servers': 'localhost:9092'}
producer = Producer(producer_conf)

def delivery_report(err, msg):
    if err is not None:
        print(f"Message delivery failed: {err}")
    else:
        print(f"Message delivered to {msg.topic()} [{msg.partition()}]")

def stream_event_to_kafka(event):
    producer.produce("snapp.events.dev", key=event["event_id"].encode('utf-8'), value=json.dumps(event).encode('utf-8'), callback=delivery_report)
    producer.poll(0)

def query_prediction_api(event_id):
    try:
        response = requests.get(f"http://localhost:8000/predict/{event_id}")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"API query failed: {e}")
        return None

# Load synthetic events
with open("synthetic_fraud_events.csv", "r") as f: # assuming CSV can be read as dict per row
    # Simplified: in reality, parse CSV to list of dicts
    import csv
    reader = csv.DictReader(f)
    events = list(reader)

for event in events[:10]: # Stream first 10 for demo
    print(f"Streaming event: {event['event_id']}")
    stream_event_to_kafka(event)
    time.sleep(1) # Simulate real-time delay
    
    # Query prediction for a processed event (requires event_id as lookup)
    # The consumer and feature store need to process it first
    # For a direct demo, you might just send the event payload to the API directly
    # For this architecture, prediction is tied to event_id after features are materialized
    
    # For a quick API demo, simulate sending processed event data
    # In reality, this would be triggered by a Kafka consumer callback
    # For this demo, we'll just query by the event_id as if it's processed
    
    # Wait for processing
    time.sleep(2) 
    prediction = query_prediction_api(event['event_id']) 
    if prediction:
        print(f"Prediction for {event['event_id']}: {prediction}")
    print("-" * 30)

producer.flush()
```
Run `python scripts/stream_and_query.py` (after creating it) or execute relevant cells in `notebooks/snapptech_fraud_demo.ipynb`.

### Step 3.5: Visualize & Interpret Fraud Decisions

The `notebooks/snapptech_fraud_demo.ipynb` notebook can be used to:
*   Visualize fraud scores over time.
*   Explore feature importance for LightGBM predictions using `src/interpretability_module/explanation_generator.py`.
*   Simulate human review and feedback.

## 4. Demonstrating MLOps Features

*   **Model Versioning**: Show different models in MLflow UI.
*   **CI/CD**: Point to GitHub Actions workflows for security checks and deployment.
*   **Monitoring**: Discuss how Prometheus/Grafana would track latency, throughput, and model metrics.
*   **Feedback Loop**: Explain how human review data updates the system.