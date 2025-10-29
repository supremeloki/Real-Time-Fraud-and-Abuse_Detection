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

        logger.info(f"Prepared {len(training_df)} samples for retraining.")
        return training_df

    def trigger_retraining(self) -> bool:
        if not self.last_retrain_timestamp:
            self._load_last_retrain_info()

        time_since_last_retrain = datetime.now() - self.last_retrain_timestamp

        if time_since_last_retrain.days < self.retrain_interval_days:
            logger.info(
                f"Retraining not due. Last retrain {time_since_last_retrain.days} days ago. Next retrain in {self.retrain_interval_days - time_since_last_retrain.days} days."
            )
            return False

        logger.info(f"Retraining triggered for {self.model_name}.")
        training_data = self._prepare_training_data()

        if training_data is None:
            logger.error("Failed to prepare training data. Retraining aborted.")
            return False

        features = training_data.drop(columns=["true_label"])
        labels = training_data["true_label"].astype(int)

        with mlflow.start_run(
            run_name=f"{self.model_name}_retrain_{datetime.now().strftime('%Y%m%d%H%M')}"
        ) as run:
            mlflow.log_param("retrain_interval_days", self.retrain_interval_days)
            mlflow.log_param("min_data_for_retrain", self.min_data_for_retrain)

            params = {
                "objective": "binary",
                "metric": "auc",
                "boosting_type": "gbdt",
                "num_leaves": 31,
                "learning_rate": 0.05,
                "feature_fraction": 0.9,
                "verbose": -1,
                "n_estimators": 100,
            }
            model = lgb.LGBMClassifier(**params)
            model.fit(features, labels)

            from sklearn.metrics import roc_auc_score, precision_score, recall_score

            predictions = model.predict_proba(features)[:, 1]
            auc = roc_auc_score(labels, predictions)
            precision = precision_score(labels, (predictions > 0.5).astype(int))
            recall = recall_score(labels, (predictions > 0.5).astype(int))

            mlflow.log_metrics(
                {"auc_score": auc, "precision": precision, "recall": recall}
            )

            mlflow.lightgbm.log_model(
                model, "lightgbm_fraud_model", registered_model_name=self.model_name
            )

            self.last_retrain_timestamp = datetime.now()
            logger.info(
                f"Model {self.model_name} successfully retrained and logged to MLflow (Run ID: {run.info.run_id})."
            )
            return True
