# src/explainability/feature_impact_monitor.py

import logging
import pandas as pd
import numpy as np
import mlflow
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from src.utils.common_helpers import (
    setup_logging,
    load_config,
)  # Assuming common_helpers.py exists
from src.data_access.data_lake_client import (
    DataLakeClient,
)  # Assuming DataLakeClient exists

logger = setup_logging(__name__)


class FeatureImpactMonitor:
    """
    Monitors changes in model feature importance (impact) over time,
    alerting on significant shifts that may indicate model drift or data quality issues.
    """

    def __init__(
        self, config_path: Path, env: str, model_name: str = "LightGBMFraudDetector"
    ):
        self.config = load_config(config_path, env)
        self.model_name = model_name
        self.mlflow_tracking_uri = self.config["environment"]["mlflow_tracking_uri"]
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)

        self.monitor_interval_hours = self.config["model_monitoring"].get(
            "feature_impact_monitor_interval_hours", 24
        )
        self.min_importance_change_threshold = self.config["model_monitoring"].get(
            "min_importance_change_threshold", 0.15
        )  # 15% change
        self.min_absolute_importance_threshold = self.config["model_monitoring"].get(
            "min_absolute_importance_threshold", 0.01
        )  # Only monitor features above 1% impact

        self.data_lake_client = DataLakeClient(self.config["data_lake_config"])
        self.feature_importance_log_prefix = self.config["data_sources"].get(
            "feature_importance_log_prefix", "model_monitoring/feature_importance"
        )

        self.last_monitor_timestamp: Optional[datetime] = None
        self.last_logged_importances: Dict[str, float] = {}

        self._load_last_monitor_info()
        logger.info(
            f"FeatureImpactMonitor initialized for {model_name}. Monitor interval: {self.monitor_interval_hours} hours."
        )

    def _load_last_monitor_info(self):
        """Loads the last recorded feature importances and timestamp."""
        # For simplicity in demo, simulate. In production, load from S3/DB.
        self.last_monitor_timestamp = datetime.now() - timedelta(
            hours=self.monitor_interval_hours + 1
        )
        # Try to load previous importances from S3
        latest_importance_df = self.data_lake_client.load_dataframe_from_s3(
            s3_prefix=self.feature_importance_log_prefix,
            s3_filename=f"{self.model_name}_latest_importance.csv",
        )
        if latest_importance_df is not None and not latest_importance_df.empty:
            self.last_logged_importances = (
                latest_importance_df.set_index("feature").iloc[0].to_dict()
            )  # Assuming single row for latest
            logger.info("Loaded last logged feature importances from S3.")
        else:
            logger.warning(
                "No previous feature importances found in S3. Starting fresh."
            )

        logger.debug(f"Simulated last monitor timestamp: {self.last_monitor_timestamp}")

    def _fetch_latest_model_feature_importances(self) -> Optional[Dict[str, float]]:
        """
        Fetches feature importances for the production model from MLflow.
        Requires the model to have been logged with feature_importances (e.g., LightGBM).
        """
        try:
            # Find the production version of the model
            client = mlflow.tracking.MlflowClient()
            model_versions = client.search_model_versions(f"name='{self.model_name}'")
            production_version = next(
                (mv for mv in model_versions if mv.current_stage == "Production"), None
            )

            if production_version:
                run_id = production_version.run_id
                # Load the model artifact directly
                # This requires that 'feature_importances_' is an attribute of the loaded model
                # or that importances were logged as a metric/artifact.

                # For LightGBM, feature_importances_ is an attribute
                # If loading via mlflow.pyfunc, you might need a custom pyfunc model wrapper
                # that exposes feature importances or logs them separately.

                # Option 1: Load model directly if it's a LightGBM model
                model_uri = f"runs:/{run_id}/lightgbm_fraud_model"
                loaded_model = mlflow.pyfunc.load_model(model_uri)._model_impl.lgb_model

                feature_importances = loaded_model.feature_importances_
                feature_names = (
                    loaded_model.feature_name_
                    if hasattr(loaded_model, "feature_name_")
                    else [f"f_{i}" for i in range(len(feature_importances))]
                )

                importance_dict = dict(zip(feature_names, feature_importances))
                logger.info(
                    f"Fetched feature importances for production model {self.model_name} (version {production_version.version})."
                )
                return importance_dict
            else:
                logger.warning(f"No production model found for {self.model_name}.")
                return None
        except Exception as e:
            logger.error(
                f"Error fetching feature importances from MLflow: {e}", exc_info=True
            )
            return None

    def monitor_feature_impact(self) -> bool:
        """
        Compares current feature importances with previous ones and logs/alerts on significant changes.
