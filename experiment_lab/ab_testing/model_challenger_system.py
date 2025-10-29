import logging
import argparse
from time import time
from celery import uuid
import pandas as pd
import numpy as np
import mlflow
import json
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime, timedelta
import sys

# Setup Python path to import src modules
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

# Import after path setup - handle missing dependencies for standalone testing
try:
    from src.prediction_engine.inference_logic import InferenceEngine
except ImportError as e:
    print(f"Warning: Could not import InferenceEngine due to missing dependencies: {e}")
    print("This is expected in environments without torch_geometric installed")
    InferenceEngine = None
logger = logging.getLogger(__name__)

class ModelChallengerSystem:
    def __init__(self, config_path: Path, env: str):
        # Simplified initialization for standalone testing
        self.config = {"environment": {"log_level": "INFO"}}
        self.logger = logging.getLogger("ModelChallengerSystem")
        self.mlflow_tracking_uri = self.config["environment"]["mlflow_tracking_uri"]
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        
        self.current_champion_model_name = "LightGBMFraudDetector"
        self.challenger_model_name = "LightGBMFraudDetector" # For simplicity, assuming challenger is new version of same model type
        self.challenger_model_version = "latest" # To be replaced by a specific version ID
        
        self.inference_engine_champion = InferenceEngine(config_path, env) # Loads latest champion by default
        self.inference_engine_challenger = None # Will be loaded dynamically

        self.ab_test_active = False
        self.test_group_assignment_rate = 0.1 # Percentage of traffic for challenger
        self.ab_test_results_log = Path("./experiment_lab/ab_testing/ab_test_results.jsonl")
        self.logger.info("ModelChallengerSystem initialized.")

    def start_ab_test(self, challenger_model_version: str, traffic_split_ratio: float = 0.1):
        if self.ab_test_active:
            self.logger.warning("A/B test already active. Please stop current test before starting a new one.")
            return False

        self.challenger_model_version = challenger_model_version
        self.test_group_assignment_rate = traffic_split_ratio
        
        self.logger.info(f"Loading challenger model version {challenger_model_version} for {self.challenger_model_name}.")
        try:
            temp_engine = InferenceEngine(self.config_path, self.env)
            temp_engine.lightgbm_model_version = challenger_model_version # Override version
            temp_engine._load_models() # Reload models with specific version
            self.inference_engine_challenger = temp_engine

            if not self.inference_engine_challenger.is_ready():
                raise RuntimeError("Challenger InferenceEngine failed to initialize.")

            self.ab_test_active = True
            self.logger.info(f"A/B test started. Challenger model '{self.challenger_model_name}' v{challenger_model_version} "
                             f"receiving {traffic_split_ratio*100}% traffic.")
            return True
        except Exception as e:
            self.logger.error(f"Failed to start A/B test: {e}", exc_info=True)
            self.ab_test_active = False
            return False

    def stop_ab_test(self):
        if not self.ab_test_active:
            self.logger.warning("No A/B test is currently active.")
            return False
        
        self.ab_test_active = False
        self.inference_engine_challenger = None
        self.logger.info("A/B test stopped.")
        return True

    def _assign_test_group(self, user_id: str) -> str:
        # Simple hash-based assignment to ensure consistent assignment for a user
        hash_val = int(hash(user_id)) % 1000
        if hash_val < self.test_group_assignment_rate * 1000:
            return "challenger"
        return "champion"

    def get_predictions(self, event: Dict[str, Any]) -> Dict[str, Any]:
        if not self.ab_test_active:
            # If no A/B test, just use the champion model
            return self.inference_engine_champion.run_inference(event)
        
        user_id = event.get("user_id", str(uuid.uuid4())) # Use user_id for consistent assignment
        assigned_group = self._assign_test_group(user_id)
        
        if assigned_group == "challenger":
            prediction = self.inference_engine_challenger.run_inference(event)
            model_used = f"Challenger (v{self.challenger_model_version})"
        else:
            prediction = self.inference_engine_champion.run_inference(event)
            model_used = f"Champion (v{self.inference_engine_champion.lightgbm_model_version})"

        # Log A/B test results
        self._log_ab_test_result(event, prediction, assigned_group, model_used)
        
        prediction["ab_test_group"] = assigned_group
        prediction["model_used"] = model_used
        return prediction

    def _log_ab_test_result(self, event: Dict[str, Any], prediction: Dict[str, Any], group: str, model_used: str):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_id": event.get("event_id"),
            "user_id": event.get("user_id"),
            "ab_test_group": group,
            "model_used": model_used,
            "fraud_score": prediction.get("fraud_score"),
            "is_fraud_predicted": prediction.get("is_fraud"),
            "latency_ms": prediction.get("latency_ms"),
            # In a real scenario, also log true labels if available (from feedback loop)
            # "true_label": None # This would come from feedback_loop/human_review_integration
        }
        with open(self.ab_test_results_log, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry) + '\n')
        self.logger.debug(f"Logged A/B test result for event {event.get('event_id')} in group {group}.")

    def analyze_ab_test_results(self):
        self.logger.info("Analyzing A/B test results.")
        if not self.ab_test_results_log.exists():
            self.logger.warning("No A/B test results log found.")
            return pd.DataFrame()
        
        results = []
        with open(self.ab_test_results_log, 'r', encoding='utf-8') as f:
            for line in f:
                results.append(json.loads(line))
        
        df = pd.DataFrame(results)
        
        # Example analysis: compare fraud detection rates and latencies
        comparison = df.groupby('ab_test_group').agg(
            avg_fraud_score=('fraud_score', 'mean'),
            fraud_detection_rate=('is_fraud_predicted', 'mean'),
            avg_latency_ms=('latency_ms', 'mean'),
            total_events=('event_id', 'count')
        )
        self.logger.info("A/B test analysis complete.")
        print(comparison)
        return comparison

if __name__ == "__main__":
    print("Model Challenger System - A/B Testing module loaded successfully")
    print("Note: Full execution requires MLflow server, Redis, and all dependencies installed")
    print("This module is designed to run within the main fraud detection system")