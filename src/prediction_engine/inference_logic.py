import argparse
import pandas as pd
import numpy as np
import logging
import time
import mlflow
import mlflow.lightgbm
import torch
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from src.utils.common_helpers import load_config, setup_logging
from src.feature_forge.real_time_features import RealTimeFeatureStore
from src.feature_forge.batch_features import BatchFeatureProcessor
from src.feature_forge.graph_features import GraphFeatureExtractor
from src.model_arsenal.train_gnn import GNNModel
from src.latency_chamber.ultra_low_latency_optimizations import optimize_inference

logger = setup_logging(__name__)


class InferenceEngine:
    def __init__(self, config_path: Path, env: str):
        self.config = load_config(config_path, env)
        self.logger = setup_logging(
            "InferenceEngine", self.config["environment"]["log_level"]
        )
        self.env = env
        self.mlflow_tracking_uri = self.config["environment"]["mlflow_tracking_uri"]
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)

        self.lightgbm_model = None
        self.gnn_model = None
        self.feature_store = RealTimeFeatureStore(config_path, env)
        self.batch_feature_processor = BatchFeatureProcessor(config_path, env)
        self.graph_feature_extractor = GraphFeatureExtractor(config_path, env)

        self.lightgbm_model_version = "latest"
        self.gnn_model_version = "latest"

        self._load_models()
        self.is_ready_flag = True
        self.logger.info("InferenceEngine initialized. Models loaded.")

    def _load_models(self):
        self.logger.info("Loading LightGBM and GNN models from MLflow.")

        # For demo purposes, skip MLflow loading and use simulation mode
        self.logger.info("Demo mode: Skipping MLflow model loading, using simulation")
        self.lightgbm_model = None
        self.gnn_model = None
        self.is_ready_flag = True
        self.logger.info(
            "InferenceEngine initialized in demo mode - models will be simulated"
        )

    @optimize_inference
    def _extract_features(self, event: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.perf_counter()
        features = {}

        user_id = event.get("user_id")
        driver_id = event.get("driver_id")
        ride_id = event.get("ride_id")
        event_time_str = event.get("event_timestamp")
        event_type = event.get("event_type")
        fare_amount = event.get("fare_amount", 0)
        distance_km = event.get("distance_km", 0)
        duration_min = event.get("duration_min", 0)

        # Basic event features
        event_time = (
            datetime.fromisoformat(event_time_str) if event_time_str else datetime.now()
        )
        features["hour_of_day"] = event_time.hour
        features["day_of_week"] = event_time.weekday()
        features["distance_per_duration"] = distance_km / (
            duration_min if duration_min > 0 else 1e-6
        )
        features["fare_per_km"] = fare_amount / (
            distance_km if distance_km > 0 else 1e-6
        )
        features["fare_amount"] = fare_amount
        features["distance_km"] = distance_km
        features["duration_min"] = duration_min

        # Real-time features from Redis
        if user_id:
            user_rt_features = self.feature_store.get_user_features(user_id)
            features.update(user_rt_features)
        if driver_id:
            driver_rt_features = self.feature_store.get_driver_features(driver_id)
            features.update(driver_rt_features)

        # Batch features (usually pre-loaded or from a different store, here we simulate)
        # In a real system, these would be retrieved from a dedicated batch feature store,
        # potentially loaded into Redis or a local cache.
        batch_user_feats_df = pd.read_csv(
            self.batch_feature_processor.data_source_path / "batch_user_features.csv"
        )
        batch_driver_feats_df = pd.read_csv(
            self.batch_feature_processor.data_source_path / "batch_driver_features.csv"
        )

        if user_id and not batch_user_feats_df.empty:
            user_batch_feats = (
                batch_user_feats_df[batch_user_feats_df["user_id"] == user_id]
                .iloc[0]
                .to_dict()
            )
            features.update(
                {k: v for k, v in user_batch_feats.items() if k != "user_id"}
            )
        if driver_id and not batch_driver_feats_df.empty:
            driver_batch_feats = (
                batch_driver_feats_df[batch_driver_feats_df["driver_id"] == driver_id]
                .iloc[0]
                .to_dict()
            )
            features.update(
                {k: v for k, v in driver_batch_feats.items() if k != "driver_id"}
            )

        # Graph features (for GNN, requires the full graph structure to be loaded or a sub-graph query)
        # For real-time GNN inference, it's not feasible to re-construct/re-train the whole graph.
        # Strategies:
        # 1. Pre-compute node embeddings (e.g., Node2Vec, GraphSAGE embeddings) in batch and store in feature store.
        # 2. Query a localized subgraph around the current user/driver and run GNN inference on that.
        # 3. Use pre-computed "graph features" (like centrality, community IDs) as flat features for LightGBM.

        # For this demo, assuming pre-computed graph features are integrated into the `features` dictionary
        # and GNN directly predicts on these (simpler, but not true GNN inference).
        # A more advanced GNN inference would involve dynamically building a mini-batch graph.

        # To truly use the GNN model trained in train_gnn.py:
        # We need the full graph or relevant subgraph, node features (x), and edge index.
        # For a single event, we'd query the graph to get the user/driver node and its neighbors.
        # Construct a 'mini-graph' Data object for the user/driver and their immediate connections.

        # Placeholder: If GNN is used for collusion, it might be triggered by a cluster of suspicious events
        # or predict for specific nodes (user/driver) based on their latest graph state.

        # For the purpose of getting a 'score', we'll make a simplified assumption:
        # The GNN model is used to predict the 'is_collusion_suspect' label for the given user/driver node
        # if such node exists and its features are passed.

        user_id = event.get("user_id")
        driver_id = event.get("driver_id")

        # This requires the GNN model's input features (x) to be consistent with the graph_features
        # and mapping the specific user/driver node to its index.
        # This is a complex step, requiring a live graph database or a cached full graph.

        # For simplicity, we'll simulate a GNN score based on the extracted graph features
        # and a heuristic for this example. In a real system, `GNNModel` would be called
        # with a subgraph and actual node features.

        gnn_score = 0.5  # Default neutral score

        if (
            user_id in self.graph_feature_extractor.load_graph_data().nodes()
        ):  # Check if node exists
            # This part requires significant architectural support for real GNN inference
            # For a demo, we will use a dummy score derived from graph features.
            # In production, this would involve creating a `torch_geometric.data.Data` object
            # for the relevant node(s) and their neighbors, passing it to the GNN model.

            # Example heuristic: if user has high graph centrality and multiple shared rides (collusion indicators)
            if (
                features.get("user_graph_degree_centrality", 0) > 0.1
                and features.get("user_unique_driver_count", 0) < 5
            ):
                gnn_score += 0.2

        if driver_id in self.graph_feature_extractor.load_graph_data().nodes():
            if (
                features.get("driver_graph_degree_centrality", 0) > 0.1
                and features.get("driver_lifetime_unique_user_count", 0) < 10
            ):
                gnn_score += 0.15

        gnn_score = np.clip(gnn_score, 0.0, 1.0)  # Ensure score is within 0-1

        end_time = time.perf_counter()
        self.logger.debug(
            f"Feature extraction took {(end_time - start_time) * 1000:.2f}ms"
        )
        return features

    @optimize_inference
    def _run_lightgbm_inference(self, features: Dict[str, Any]) -> float:
        start_time = time.perf_counter()

        if self.lightgbm_model is None:
            # Demo mode: simulate LightGBM prediction based on feature heuristics
            score = self._simulate_lightgbm_score(features)
            end_time = time.perf_counter()
            self.logger.debug(
                f"LightGBM inference (simulated) took {(end_time - start_time) * 1000:.2f}ms"
            )
            return score

        # Align features with training features for LightGBM
        # This assumes the LightGBM model internally stores its feature names
        # Or, we define a fixed set of features for inference

        # This list of features MUST match the features used in src/model_arsenal/train_lightgbm.py
        # For simplicity, we'll extract them dynamically for demo, but typically this is fixed.
        model_features = (
            self.lightgbm_model.feature_name_
            if hasattr(self.lightgbm_model, "feature_name_")
            else list(features.keys())
        )

        input_df = pd.DataFrame([features])[model_features]
        # Ensure all columns expected by the model are present, fill missing with 0 or a reasonable default
        for col in model_features:
            if col not in input_df.columns:
                input_df[col] = 0  # Or some other imputation strategy

        probability = self.lightgbm_model.predict_proba(input_df)[:, 1][0]
        end_time = time.perf_counter()
        self.logger.debug(
            f"LightGBM inference took {(end_time - start_time) * 1000:.2f}ms"
        )
        return float(probability)

    def _simulate_lightgbm_score(self, features: Dict[str, Any]) -> float:
        """Simulate LightGBM prediction for demo purposes when model is not available"""
        # Simple heuristic-based scoring for demonstration
        base_score = 0.1

        # Fraud indicators
        if features.get("fare_amount", 0) < 50000:  # Very low fare
            base_score += 0.3
        if features.get("distance_km", 0) < 2:  # Very short distance
            base_score += 0.2
        if features.get("duration_min", 0) < 5:  # Very short duration
            base_score += 0.2
        if features.get("hour_of_day", 12) in [1, 2, 3, 4, 5]:  # Late night rides
            base_score += 0.1

        # Normalize to 0-1 range
        return min(max(base_score, 0.0), 1.0)

    @optimize_inference
    def _run_gnn_inference(
        self, event: Dict[str, Any], features: Dict[str, Any]
    ) -> float:
        start_time = time.perf_counter()
        # For real-time GNN inference, it's not feasible to re-construct/re-train the whole graph.
        # Strategies:
        # 1. Pre-compute node embeddings (e.g., Node2Vec, GraphSAGE embeddings) in batch and store in feature store.
        # 2. Query a localized subgraph around the current user/driver and run GNN inference on that.
        # 3. Use pre-computed "graph features" (like centrality, community IDs) as flat features for LightGBM.

        # For this demo, assuming pre-computed graph features are integrated into the `features` dictionary
        # and GNN directly predicts on these (simpler, but not true GNN inference).
        # A more advanced GNN inference would involve dynamically building a mini-batch graph.

        # To truly use the GNN model trained in train_gnn.py:
        # We need the full graph or relevant subgraph, node features (x), and edge index.
        # For a single event, we'd query the graph to get the user/driver node and its neighbors.
        # Construct a 'mini-graph' Data object for the user/driver and their immediate connections.

        # Placeholder: If GNN is used for collusion, it might be triggered by a cluster of suspicious events
        # or predict for specific nodes (user/driver) based on their latest graph state.

        # For the purpose of getting a 'score', we'll make a simplified assumption:
        # The GNN model is used to predict the 'is_collusion_suspect' label for the given user/driver node
        # if such node exists and its features are passed.

        user_id = event.get("user_id")
        driver_id = event.get("driver_id")

        # This requires the GNN model's input features (x) to be consistent with the graph_features
        # and mapping the specific user/driver node to its index.
        # This is a complex step, requiring a live graph database or a cached full graph.

        # For simplicity, we'll simulate a GNN score based on the extracted graph features
        # and a heuristic for this example. In a real system, `GNNModel` would be called
        # with a subgraph and actual node features.

        gnn_score = 0.5  # Default neutral score

        if (
            user_id in self.graph_feature_extractor.load_graph_data().nodes()
        ):  # Check if node exists
            # This part requires significant architectural support for real GNN inference
            # For a demo, we will use a dummy score derived from graph features.
            # In production, this would involve creating a `torch_geometric.data.Data` object
            # for the relevant node(s) and their neighbors, passing it to the GNN model.

            # Example heuristic: if user has high graph centrality and multiple shared rides (collusion indicators)
            if (
                features.get("user_graph_degree_centrality", 0) > 0.1
                and features.get("user_unique_driver_count", 0) < 5
            ):
                gnn_score += 0.2

        if driver_id in self.graph_feature_extractor.load_graph_data().nodes():
            if (
                features.get("driver_graph_degree_centrality", 0) > 0.1
                and features.get("driver_lifetime_unique_user_count", 0) < 10
            ):
                gnn_score += 0.15

        gnn_score = np.clip(gnn_score, 0.0, 1.0)  # Ensure score is within 0-1

        end_time = time.perf_counter()
        self.logger.debug(
            f"GNN inference (simulated) took {(end_time - start_time) * 1000:.2f}ms"
        )
        return float(gnn_score)

    def run_inference(self, event: Dict[str, Any]) -> Dict[str, Any]:
        overall_start_time = time.perf_counter()

        features = self._extract_features(event)

        lightgbm_score = self._run_lightgbm_inference(features)
        gnn_score = self._run_gnn_inference(
            event, features
        )  # Pass event for context if needed

        # Ensemble strategy
        ensemble_config = self.config["model_config"]["ensemble_strategy"]
        if ensemble_config["method"] == "weighted_average":
            final_fraud_score = (
                lightgbm_score * ensemble_config["lightgbm_weight"]
                + gnn_score * ensemble_config["gnn_weight"]
            )
        else:
            final_fraud_score = lightgbm_score  # Fallback

        threshold = self.config["thresholds"]["detection_thresholds"][
            "lightgbm_fraud_score"
        ][
            "high"
        ]  # Example threshold
        is_fraud = final_fraud_score >= threshold

        action_recommended = "monitor"
        if (
            final_fraud_score
            >= self.config["thresholds"]["action_triggers"]["auto_block_score"]
        ):
            action_recommended = "auto_block"
        elif (
            final_fraud_score
            >= self.config["thresholds"]["action_triggers"]["manual_review_queue_score"]
        ):
            action_recommended = "manual_review"

        overall_latency_ms = (time.perf_counter() - overall_start_time) * 1000

        explanation = {}  # Placeholder for actual explanation

        return {
            "is_fraud": is_fraud,
            "fraud_score": final_fraud_score,
            "model_version": f"LGBM:{self.lightgbm_model_version},GNN:{self.gnn_model_version}",
            "explanation": explanation,
            "action_recommended": action_recommended,
            "latency_ms": overall_latency_ms,
        }

    def is_ready(self) -> bool:
        return self.is_ready_flag and (
            self.lightgbm_model is not None or self.gnn_model is not None
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Inference Engine")
    parser.add_argument(
        "--env", type=str, default="dev", help="Environment (dev or prod)"
    )
    args = parser.parse_args()

    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    config_directory = project_root / "config"

    engine = InferenceEngine(config_directory, args.env)

    # Example event
    test_event = {
        "event_id": "test_event_12345",
        "event_timestamp": datetime.now().isoformat(),
        "event_type": "ride_completed",
        "user_id": "test_user_123",
        "driver_id": "test_driver_456",
        "ride_id": "test_ride_789",
        "start_location_lat": 35.72,
        "start_location_lon": 51.42,
        "end_location_lat": 35.73,
        "end_location_lon": 51.43,
        "fare_amount": 75000.0,
        "distance_km": 8.5,
        "duration_min": 25.0,
        "payment_method": "credit_card",
        "promo_code_used": None,
        "device_info": "android_12",
        "ip_address": "192.168.1.100",
    }

    if engine.is_ready():
        prediction = engine.run_inference(test_event)
        print(f"Prediction result: {prediction}")
    else:
        logger.error("Inference engine not ready. Check model loading or dependencies.")
