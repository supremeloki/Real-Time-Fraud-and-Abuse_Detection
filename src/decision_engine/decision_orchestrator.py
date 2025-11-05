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
