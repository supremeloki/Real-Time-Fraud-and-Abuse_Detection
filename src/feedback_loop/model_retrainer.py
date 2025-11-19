import logging
import pandas as pd
import numpy as np
import mlflow
import lightgbm as lgb
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from src.utils.common_helpers import load_config, setup_logging

logger = setup_logging(__name__)


class ModelRetrainer:
    def __init__(
        self, config_path: Path, env: str, model_name: str = "LightGBMFraudDetector"
    ):
        self.config = load_config(config_path, env)
        self.model_name = model_name
        self.mlflow_tracking_uri = self.config["environment"]["mlflow_tracking_uri"]
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)

        self.retrain_interval_days = self.config["model_retraining"]["interval_days"]
        self.min_data_for_retrain = self.config["model_retraining"]["min_data_points"]
        self.data_source_path = Path(self.config["data_sources"]["training_data_path"])
        self.feedback_data_path = Path(
            self.config["data_sources"]["human_feedback_path"]
        )
        self.last_retrain_timestamp: Optional[datetime] = None
        self._load_last_retrain_info()
        logger.info(
            f"ModelRetrainer initialized for {model_name}. Retrain interval: {self.retrain_interval_days} days."
        )

    def _load_last_retrain_info(self):
        self.last_retrain_timestamp = datetime.now() - timedelta(
            days=self.retrain_interval_days + 1
        )
        logger.debug(f"Simulated last retrain timestamp: {self.last_retrain_timestamp}")

    def _prepare_training_data(self) -> Optional[pd.DataFrame]:
        if not self.data_source_path.exists():
            logger.error(f"Training data source not found: {self.data_source_path}")
            return None

        training_df = pd.read_csv(self.data_source_path)

        if self.feedback_data_path.exists():
            feedback_df = pd.read_json(self.feedback_data_path, lines=True)
            feedback_df = feedback_df[["event_id", "human_decision"]].rename(
                columns={"human_decision": "true_label"}
            )
            training_df = training_df.merge(feedback_df, on="event_id", how="left")
            training_df["true_label"] = training_df["true_label"].fillna(
                training_df["is_fraud_scenario"]
            )
        else:
            training_df["true_label"] = training_df["is_fraud_scenario"]

        feature_columns = ["fare_amount", "distance_km", "duration_min", "hour_of_day"]

        for col in feature_columns:
            if col not in training_df.columns:
                logger.warning(
                    f"Feature '{col}' not found in training data. Filling with 0."
                )
                training_df[col] = 0.0

        training_df = training_df[feature_columns + ["true_label"]]
        training_df = training_df.dropna()

        if len(training_df) < self.min_data_for_retrain:
            logger.warning(
                f"Not enough data for retraining. Required: {self.min_data_for_retrain}, available: {len(training_df)}"
            )
            return None

