# src/decision_engine/decision_orchestrator.py

import logging
import json
import time
import lightgbm as lgb
import pandas as pd
import numpy as np

from typing import Dict, Any, Tuple, Callable, List, Optional
from datetime import datetime
from sklearn.datasets import make_classification

# Add project root to Python path for imports
import sys
from pathlib import Path

current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.common_helpers import setup_logging, load_config
from src.feature_forge.realtime_feature_engineer import RealtimeFeatureEngineer
from src.graph_processor.node_embedding_updater import NodeEmbeddingUpdater
from src.graph_processor.temporal_graph_analyzer import TemporalGraphAnalyzer
from src.graph_processor.graph_anomaly_detector import GraphAnomalyDetector
from src.profile_builder.user_behavioral_profiler import UserBehavioralProfiler
from src.data_access.feature_store_client import FeatureStoreClient
from src.security.threat_intelligence_feed import ThreatIntelligenceFeed
from src.risk_scoring.dynamic_risk_policy_engine import DynamicRiskPolicyEngine
from src.experiment_engine.ab_test_manager import ABTestManager
from src.explainability.shap_explainer import (
    SHAPExplainer,
)  # Assuming it's initialized externally with a model
from src.data_quality.stream_data_validator import StreamDataValidator
from src.monitoring.operational_metrics_collector import OperationalMetricsCollector
from src.data_access.data_lake_client import DataLakeClient

logger = logging.getLogger(__name__)


class DecisionOrchestrator:
    def __init__(
        self,
        config_path: str,
        env: str,
        model_inference_func: Callable[[pd.DataFrame], np.ndarray],
        model_feature_names: List[str],
        model_training_data_sample: pd.DataFrame,
    ):

        self.config = load_config(config_path, env)

        # Initialize sub-components
        self.feature_store = FeatureStoreClient(self.config)
        self.threat_intel = ThreatIntelligenceFeed(
            self.config.get("threat_intel_config", {})
        )
        self.behavioral_profiler = UserBehavioralProfiler(self.config["redis_config"])
        self.temporal_graph_analyzer = TemporalGraphAnalyzer(
            self.config["redis_config"]
        )
        self.graph_anomaly_detector = GraphAnomalyDetector(self.config["redis_config"])
        self.policy_engine = DynamicRiskPolicyEngine(
            self.config.get("risk_policy_config", {})
        )
        self.ab_test_manager = ABTestManager(self.config.get("ab_test_config", {}))
        self.data_validator = StreamDataValidator(
            self.config.get("data_validation_schema", {})
        )
        self.metrics_collector = OperationalMetricsCollector()

        # SHAP explainer requires a trained model and training data sample
        self.shap_explainer = SHAPExplainer(
            model=model_inference_func.__self__,  # Assumes model_inference_func is a bound method, e.g., model.predict_proba
            feature_names=model_feature_names,
            training_data_sample=model_training_data_sample,
            explainer_type="tree",  # or 'kernel' based on model type
        )
        self.model_inference_func = model_inference_func

        logger.info("DecisionOrchestrator initialized, ready to process events.")

    def process_event(self, raw_event: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.perf_counter()
        event_id = raw_event.get("event_id", "UNKNOWN")
        logger.info(f"Processing event: {event_id}")

        # 1. Data Validation
        is_valid, validation_issues = self.data_validator.validate_event(raw_event)
        if not is_valid:
            self.metrics_collector.record_error(
                "DataValidationFailure",
                {"event_id": event_id, "issues": validation_issues},
            )
            logger.error(
                f"Event {event_id} failed validation. Skipping further processing."
            )
            return {
                "event_id": event_id,
                "status": "rejected",
                "reason": "Data validation failed",
                "issues": validation_issues,
            }

        current_timestamp = datetime.now()

        # 2. Update and Retrieve Features
        try:
            # Threat intelligence features
            threat_intel_features = self.threat_intel.check_event_for_threats(raw_event)

            # User behavioral profile updates and features
            self.behavioral_profiler.update_user_profile(raw_event)
            behavioral_features = self.behavioral_profiler.get_user_behavioral_features(
                raw_event.get("user_id"), current_timestamp
            )

            # Temporal Graph updates and features
            self.temporal_graph_analyzer.update_temporal_graph_state(raw_event)
            temporal_features = self.temporal_graph_analyzer.analyze_temporal_patterns(
                raw_event.get("user_id"), raw_event.get("driver_id"), current_timestamp
            )

            # Aggregate all features
            all_features = self.feature_store.get_features_for_event(
                raw_event, current_timestamp
            )
            all_features.update(threat_intel_features)
            all_features.update(behavioral_features)
            all_features.update(temporal_features)

        except Exception as e:
            self.metrics_collector.record_error(
                "FeatureEngineeringFailure", {"event_id": event_id, "error": str(e)}
            )
            logger.error(
                f"Feature engineering failed for event {event_id}: {e}", exc_info=True
            )
            return {
                "event_id": event_id,
                "status": "error",
                "reason": "Feature engineering failed",
            }

        # 3. Model Inference
        try:
            # Prepare feature vector for model
            model_input_df = pd.DataFrame([all_features])[
                self.shap_explainer.feature_names
            ]
            for (
                col
            ) in self.shap_explainer.feature_names:  # Ensure all features are present
                if col not in model_input_df.columns:
                    model_input_df[col] = 0.0  # Or a sensible default
            model_input_df = model_input_df[
                self.shap_explainer.feature_names
            ]  # Ensure order

            prediction_proba = self.model_inference_func(model_input_df)[0]
            fraud_score = float(prediction_proba[1])
            is_fraud_predicted = (
                fraud_score
                >= self.config["thresholds"]["detection_thresholds"][
                    "lightgbm_fraud_score"
                ]["high"]
            )  # Default threshold

            model_prediction = {
                "event_id": event_id,
                "fraud_score": fraud_score,
                "is_fraud": is_fraud_predicted,
                "action_recommended": (
                    "auto_block"
                    if fraud_score
                    >= self.config["thresholds"]["action_triggers"]["auto_block_score"]
                    else (
                        "manual_review"
                        if fraud_score
                        >= self.config["thresholds"]["detection_thresholds"][
                            "lightgbm_fraud_score"
                        ]["medium"]
                        else "monitor"
                    )
                ),
            }
            logger.info(
                f"Model prediction for {event_id}: Fraud Score={fraud_score:.4f}, Action={model_prediction['action_recommended']}"
            )
            self.metrics_collector.increment_throughput()
        except Exception as e:
            self.metrics_collector.record_error(
                "ModelInferenceFailure", {"event_id": event_id, "error": str(e)}
            )
            logger.error(
                f"Model inference failed for event {event_id}: {e}", exc_info=True
            )
            return {
                "event_id": event_id,
                "status": "error",
                "reason": "Model inference failed",
            }

        # 4. Dynamic Risk Policy Evaluation
        try:
            policy_results = self.policy_engine.evaluate_policies(
                all_features, model_prediction
            )
            final_action = policy_results["final_action_with_model"]
            triggered_policies = policy_results["triggered_policies"]
            logger.info(
                f"Policy engine decision for {event_id}: Final Action={final_action}, Triggered Policies={triggered_policies}"
            )
            model_prediction["action_recommended"] = (
                final_action  # Update recommended action
            )
            model_prediction["triggered_policies"] = triggered_policies
        except Exception as e:
            self.metrics_collector.record_error(
                "PolicyEngineFailure", {"event_id": event_id, "error": str(e)}
            )
            logger.error(
                f"Policy engine failed for event {event_id}: {e}", exc_info=True
            )

        # 5. Graph Anomaly Detection
        try:
            graph_anomalies = self.graph_anomaly_detector.analyze_graph_for_anomalies(
                current_timestamp
            )
            if graph_anomalies:
                model_prediction["graph_anomalies"] = graph_anomalies
                # Optionally escalate action if critical graph anomalies are found
                if (
                    "user_high_degree_centrality_anomaly" in graph_anomalies
                    or "dense_small_community_detected" in graph_anomalies
                ):
                    if model_prediction["action_recommended"] == "monitor":
                        model_prediction["action_recommended"] = "manual_review"
                logger.warning(
                    f"Graph anomalies detected for {event_id}: {json.dumps(graph_anomalies)}"
                )
        except Exception as e:
            self.metrics_collector.record_error(
                "GraphAnomalyDetectionFailure", {"event_id": event_id, "error": str(e)}
            )
            logger.error(
                f"Graph anomaly detection failed for event {event_id}: {e}",
                exc_info=True,
            )

        # 6. A/B Testing Integration
        try:
            ab_assignments = self.ab_test_manager.get_experiment_assignment(
                raw_event.get("user_id") or raw_event.get("event_id"), raw_event
            )
            model_prediction["ab_assignments"] = ab_assignments
            if ab_assignments:
                logger.info(f"Event {event_id} part of A/B tests: {ab_assignments}")
        except Exception as e:
            self.metrics_collector.record_error(
                "ABTestFailure", {"event_id": event_id, "error": str(e)}
            )
            logger.error(
                f"A/B test assignment failed for event {event_id}: {e}", exc_info=True
            )

        # 7. Explainability (SHAP values for important decisions)
        try:
            if model_prediction["action_recommended"] in [
                "auto_block",
                "manual_review",
            ]:
                shap_explanation = self.shap_explainer.explain_instance(all_features)
                model_prediction["explanation"] = shap_explanation
                logger.info(f"Generated SHAP explanation for event {event_id}.")
        except Exception as e:
            self.metrics_collector.record_error(
                "SHAPExplanationFailure", {"event_id": event_id, "error": str(e)}
            )
            logger.error(
                f"SHAP explanation failed for event {event_id}: {e}", exc_info=True
            )

        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000
        self.metrics_collector.record_latency(latency_ms)
        logger.info(f"Event {event_id} processed in {latency_ms:.2f}ms.")

        return model_prediction


if __name__ == "__main__":
    print("DecisionOrchestrator - Module loaded successfully")
    print("Note: Full execution requires Redis, MLflow, and all dependencies installed")
    print("This module is designed to run within the main fraud detection system")
    exit(0)  # Exit gracefully since this is a library module

    # Setup dummy environment and files for testing
    project_root = Path(__file__).parent.parent.parent
    config_directory = project_root / "config"
    data_vault_directory = project_root / "data_vault"

    # Ensure dummy config files exist for load_config to work
    (config_directory / "environments").mkdir(parents=True, exist_ok=True)
    with open(config_directory / "environments" / "dev.yaml", "w") as f:
        f.write(
            "environment:\n  mlflow_tracking_uri: 'file://./mlruns_demo'\n  log_level: 'INFO'\ndata_lake_config:\n  s3_bucket_name: 'test-bucket'\n  aws_region: 'us-east-1'\nredis_config:\n  redis_host: 'localhost'\n  redis_port: 6379\n  redis_db: 0\n  default_ttl_seconds: 3600\nrealtime_feature_windows: [1, 5]\ngnn_embedding_dimension: 4\nthreat_intel_config:\n  ip_blacklist_path: 'data_vault/threat_intel/ip_blacklist.json'\n  device_blacklist_path: 'data_vault/threat_intel/device_blacklist.json'\n  promo_watchlist_path: 'data_vault/threat_intel/promo_watchlist.json'\nrisk_policy_config:\n  policies:\n    - name: 'Blacklisted IP Auto-Block'\n      conditions:\n        - feature: 'is_ip_blacklisted'\n          operator: '=='\n          value: True\n      action: 'block'\n      priority: 100\n    - name: 'High Fraud Score Review'\n      conditions:\n        - feature: 'fraud_score'\n          operator: '>='\n          value: 0.7\n      action: 'review'\n      priority: 50\n  default_action: 'monitor'\nab_test_config:\n  experiments:\n    model_v1_vs_v2:\n      name: 'Model_V1_vs_V2_Fraud_Detection'\n      start_time: '2025-10-18T00:00:00'\n      end_time: '2025-11-18T00:00:00'\n      total_traffic_percentage: 0.5\n      variants:\n        control_v1: {traffic_percentage: 0.5}\n        test_v2: {traffic_percentage: 0.5}\ndata_validation_schema:\n  event_id: {required: True, type: 'string', regex_pattern: '^e[0-9]{3}$'}\n  fare_amount: {required: True, type: 'float', min_value: 100, max_value: 1000000}\n  user_id: {required: True, type: 'string'}\nthresholds:\n  detection_thresholds:\n    lightgbm_fraud_score:\n      high: 0.75\n      medium: 0.5\n  action_triggers:\n    auto_block_score: 0.9"
        )

    # Create dummy data_vault folders and files
    (data_vault_directory / "batch_features").mkdir(parents=True, exist_ok=True)
    (data_vault_directory / "graph_topology_data").mkdir(parents=True, exist_ok=True)
    (data_vault_directory / "threat_intel").mkdir(parents=True, exist_ok=True)

    pd.DataFrame({"user_id": ["u1", "u2"], "user_lifetime_rides": [100, 50]}).to_csv(
        data_vault_directory / "batch_features" / "batch_user_features.csv", index=False
    )
    pd.DataFrame(
        {"driver_id": ["d1", "d2"], "driver_lifetime_rides": [200, 120]}
    ).to_csv(
        data_vault_directory / "batch_features" / "batch_driver_features.csv",
        index=False,
    )
    pd.DataFrame(
        {
            "node_id": ["u1", "d1"],
            "node_type": ["user", "driver"],
            "degree_centrality": [0.5, 0.7],
        }
    ).to_csv(
        data_vault_directory / "graph_topology_data" / "processed_graph_features.csv",
        index=False,
    )

    with open(data_vault_directory / "threat_intel" / "ip_blacklist.json", "w") as f:
        json.dump(["192.168.1.100"], f)
    with open(
        data_vault_directory / "threat_intel" / "device_blacklist.json", "w"
    ) as f:
        json.dump([], f)
    with open(data_vault_directory / "threat_intel" / "promo_watchlist.json", "w") as f:
        json.dump([], f)

    # Mock Model and Feature Names for SHAP
    X, y = make_classification(
        n_samples=100, n_features=15, n_informative=10, n_redundant=0, random_state=42
    )
    model_feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    # Add some domain-specific feature names that we expect to see
    model_feature_names[0] = "fare_amount"
    model_feature_names[1] = "distance_km"
    model_feature_names[2] = "user_avg_fare_5min"
    model_feature_names[3] = "driver_avg_fare_5min"
    model_feature_names[4] = "user_batch_user_lifetime_rides"  # Example batch feature
    model_feature_names[5] = "is_ip_blacklisted"  # Example threat intel feature

    mock_training_df = pd.DataFrame(X, columns=model_feature_names)
    mock_training_df["is_fraud"] = y

    mock_model = lgb.LGBMClassifier(n_estimators=10, random_state=42)
    mock_model.fit(mock_training_df[model_feature_names], mock_training_df["is_fraud"])

    def mock_predict_proba_func(data: pd.DataFrame) -> np.ndarray:
        return mock_model.predict_proba(data[model_feature_names])

    # Instantiate the Orchestrator
    orchestrator = DecisionOrchestrator(
        config_path=str(config_directory),
        env="dev",
        model_inference_func=mock_predict_proba_func,
        model_feature_names=model_feature_names,
        model_training_data_sample=mock_training_df.sample(20),
    )

    # Mock DataLakeClient to use local files for demo in FeatureStoreClient
    class MockDataLakeClientDC(DataLakeClient):
        def __init__(self, config_data):
            super().__init__(config_data)
            self.s3_bucket = "dummy-bucket"
            self.base_path = data_vault_directory
            logger.info("MockDataLakeClientDC initialized for local file access.")

        def load_dataframe_from_s3(
            self, s3_prefix: str, s3_filename: str, file_format: str = "csv"
        ) -> Optional[pd.DataFrame]:
            full_path = self.base_path / s3_prefix / s3_filename
            if full_path.exists():
                logger.debug(f"Mock loading from local path: {full_path}")
                return pd.read_csv(full_path)
            logger.warning(f"Mock file not found: {full_path}")
            return None

        def save_dataframe_to_s3(
            self,
            df: pd.DataFrame,
            s3_prefix: str,
            s3_filename: str,
            file_format: str = "csv",
        ) -> bool:
            target_dir = self.base_path / s3_prefix
            target_dir.mkdir(parents=True, exist_ok=True)
            full_path = target_dir / s3_filename
            df.to_csv(full_path, index=False)
            logger.debug(f"Mock saving DataFrame to local path: {full_path}.")
            return True

    orchestrator.feature_store.data_lake_client = MockDataLakeClientDC(
        orchestrator.config["data_lake_config"]
    )

    # Test Redis connection for sub-components
    try:
        orchestrator.feature_store.redis_client.check_connection()
    except Exception as e:
        print(
            f"Could not connect to Redis: {e}. Please ensure Redis is running on localhost:6379 for this demo."
        )
        exit()

    # --- Simulate Incoming Events ---
    print("\n--- Processing various simulated events ---")

    # Event 1: Normal event
    event1 = {
        "event_id": "e001",
        "event_timestamp": datetime.now().isoformat(),
        "user_id": "u1",
        "driver_id": "d1",
        "fare_amount": 75000.0,
        "distance_km": 15.0,
        "duration_min": 30.0,
        "ip_address": "192.168.1.1",
        "payment_method": "credit_card",
    }
    result1 = orchestrator.process_event(event1)
    print(f"\nResult for {event1['event_id']}:\n{json.dumps(result1, indent=2)}")

    # Event 2: High fraud score, possibly triggers review
    event2 = {
        "event_id": "e002",
        "event_timestamp": datetime.now().isoformat(),
        "user_id": "u2",
        "driver_id": "d2",
        "fare_amount": 10000.0,
        "distance_km": 2.0,
        "duration_min": 5.0,
        "ip_address": "10.0.0.10",
        "payment_method": "cash",
        "promo_code_used": "FREE_RIDE",
    }
    result2 = orchestrator.process_event(event2)
    print(f"\nResult for {event2['event_id']}:\n{json.dumps(result2, indent=2)}")

    # Event 3: Blacklisted IP (should be blocked by policy)
    event3 = {
        "event_id": "e003",
        "event_timestamp": datetime.now().isoformat(),
        "user_id": "u1",
        "driver_id": "d1",
        "fare_amount": 120000.0,
        "distance_km": 25.0,
        "duration_min": 40.0,
        "ip_address": "192.168.1.100",
        "payment_method": "wallet",
    }
    result3 = orchestrator.process_event(event3)
    print(f"\nResult for {event3['event_id']}:\n{json.dumps(result3, indent=2)}")

    # Event 4: Data validation failure
    event4 = {
        "event_id": "invalid_event_004",
        "event_timestamp": datetime.now().isoformat(),
        "user_id": "u3",
        "driver_id": "d1",
        "fare_amount": 50.0,
        "distance_km": "invalid_distance",
        "payment_method": "credit_card",  # Invalid fare_amount and distance_km
    }
    result4 = orchestrator.process_event(event4)
    print(f"\nResult for {event4['event_id']}:\n{json.dumps(result4, indent=2)}")

    print("\n--- Current Operational Metrics ---")
    print(json.dumps(orchestrator.metrics_collector.get_aggregated_metrics(), indent=2))

    # Clean up dummy files and folders
    import shutil

    shutil.rmtree(config_directory / "environments", ignore_errors=True)
    (data_vault_directory / "batch_features" / "batch_user_features.csv").unlink(
        missing_ok=True
    )
    (data_vault_directory / "batch_features" / "batch_driver_features.csv").unlink(
        missing_ok=True
    )
    (
        data_vault_directory / "graph_topology_data" / "processed_graph_features.csv"
    ).unlink(missing_ok=True)
    (data_vault_directory / "threat_intel" / "ip_blacklist.json").unlink(
        missing_ok=True
    )
    (data_vault_directory / "threat_intel" / "device_blacklist.json").unlink(
        missing_ok=True
    )
    (data_vault_directory / "threat_intel" / "promo_watchlist.json").unlink(
        missing_ok=True
    )
    (data_vault_directory / "batch_features").rmdir()
    (data_vault_directory / "graph_topology_data").rmdir()
    (data_vault_directory / "threat_intel").rmdir()
    print("\nDemo cleanup complete.")
