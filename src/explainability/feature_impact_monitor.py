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
        """
        if not self.last_monitor_timestamp:
            self._load_last_monitor_info()

        time_since_last_monitor = datetime.now() - self.last_monitor_timestamp

        if time_since_last_monitor.total_seconds() / 3600 < self.monitor_interval_hours:
            logger.info(
                f"Feature impact monitoring not due. Last run {time_since_last_monitor.total_seconds() / 3600:.1f} hours ago. Next run in {self.monitor_interval_hours - time_since_last_monitor.total_seconds() / 3600:.1f} hours."
            )
            return False

        logger.info("Triggering feature impact monitoring.")
        current_importances = self._fetch_latest_model_feature_importances()

        if not current_importances:
            logger.warning(
                "Could not retrieve current feature importances. Aborting monitoring."
            )
            return False

        if not self.last_logged_importances:
            # First run or no history, just log current and save
            logger.info(
                "No historical importances for comparison. Logging current importances."
            )
            self.last_logged_importances = current_importances
            self._save_current_importances_to_s3(current_importances)
            self.last_monitor_timestamp = datetime.now()
            return True

        # Compare current vs. last logged importances
        significant_changes = {}
        for feature, current_impact in current_importances.items():
            last_impact = self.last_logged_importances.get(
                feature, 0.0
            )  # Default to 0 if feature is new

            # Only consider features with non-negligible impact
            if (
                current_impact < self.min_absolute_importance_threshold
                and last_impact < self.min_absolute_importance_threshold
            ):
                continue

            if (
                last_impact == 0
                and current_impact > self.min_absolute_importance_threshold
            ):
                # New important feature
                change_percent = 1.0  # Or some large number
                significant_changes[feature] = {
                    "change_percent": change_percent,
                    "from": last_impact,
                    "to": current_impact,
                    "type": "newly_important",
                }
            elif last_impact > 0:
                change_percent = abs(current_impact - last_impact) / last_impact
                if change_percent >= self.min_importance_change_threshold:
                    significant_changes[feature] = {
                        "change_percent": change_percent,
                        "from": last_impact,
                        "to": current_impact,
                        "type": "significant_shift",
                    }

        if significant_changes:
            logger.warning(
                f"Significant feature impact shifts detected: {json.dumps(significant_changes, indent=2)}"
            )
            # In a real system, send alert to ML Ops team
            with mlflow.start_run(
                run_name=f"feature_impact_alert_{datetime.now().strftime('%Y%m%d%H%M')}"
            ) as run:
                mlflow.log_dict(significant_changes, "feature_impact_changes.json")
                mlflow.log_param("alert_timestamp", datetime.now().isoformat())
                mlflow.log_param("model_name", self.model_name)
                logger.info(
                    f"Logged feature impact alert to MLflow (Run ID: {run.info.run_id})."
                )
        else:
            logger.info("No significant feature impact changes detected.")

        self.last_logged_importances = current_importances
        self._save_current_importances_to_s3(current_importances)
        self.last_monitor_timestamp = datetime.now()
        return True

    def _save_current_importances_to_s3(self, importances: Dict[str, float]):
        """Saves the current feature importances to S3."""
        df_importances = pd.DataFrame(
            list(importances.items()), columns=["feature", "importance"]
        )
        df_importances["timestamp"] = datetime.now().isoformat()

        # Save as the "latest" and also versioned for history
        self.data_lake_client.save_dataframe_to_s3(
            df_importances,
            self.feature_importance_log_prefix,
            f"{self.model_name}_latest_importance.csv",
        )
        self.data_lake_client.save_dataframe_to_s3(
            df_importances,
            self.feature_importance_log_prefix,
            f"{self.model_name}_importance_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv",
        )
        logger.info(
            f"Current feature importances saved to S3 at '{self.feature_importance_log_prefix}'."
        )


if __name__ == "__main__":
    import os
    import lightgbm as lgb
    from sklearn.datasets import make_classification

    # Setup dummy environment and files for testing
    project_root = Path(__file__).parent.parent.parent
    config_directory = project_root / "config"
    data_vault_directory = project_root / "data_vault"
    mlruns_dir = project_root / "mlruns_feature_monitor_demo"

    # Ensure dummy config files exist for load_config to work
    (config_directory / "environments").mkdir(parents=True, exist_ok=True)
    with open(config_directory / "environments" / "dev.yaml", "w") as f:
        f.write(
            f"environment:\n  mlflow_tracking_uri: 'file://{mlruns_dir}'\n  log_level: 'INFO'\ndata_lake_config:\n  s3_bucket_name: 'test-bucket'\n  aws_region: 'us-east-1'\nmodel_monitoring:\n  feature_impact_monitor_interval_hours: 0.01 # Very short for demo\n  min_importance_change_threshold: 0.1\n  min_absolute_importance_threshold: 0.01\ndata_sources:\n  feature_importance_log_prefix: 'model_monitoring/feature_importance'\n"
        )

    # Mock DataLakeClient to use local files directly for demo (no S3 interaction)
    class MockDataLakeClientFI(DataLakeClient):
        def __init__(self, config_data):
            super().__init__(config_data)
            self.s3_bucket = "dummy-bucket"  # Override to prevent actual S3 calls
            self.base_path = data_vault_directory  # Simulate S3 prefix locally
            logger.info(
                "MockDataLakeClientFI initialized, will simulate S3 interaction locally."
            )

        def load_dataframe_from_s3(
            self, s3_prefix: str, s3_filename: str, file_format: str = "csv"
        ) -> Optional[pd.DataFrame]:
            full_path = self.base_path / s3_prefix / s3_filename
            if full_path.exists():
                logger.info(f"Mock loading from local path: {full_path}")
                if file_format == "csv":
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
            if file_format == "csv":
                df.to_csv(full_path, index=False)
                logger.info(f"Mock saving DataFrame to local path: {full_path}.")
                return True
            logger.warning(f"Mock saving failed: unsupported format {file_format}.")
            return False

    # Initialize MLflow for demo
    mlflow.set_tracking_uri(f"file://{mlruns_dir}")

    # 1. Train a dummy LightGBM model and log it to MLflow
    X, y = make_classification(
        n_samples=100, n_features=5, n_informative=3, n_redundant=0, random_state=42
    )
    feature_names = [f"mock_feature_{i}" for i in range(X.shape[1])]
    df_train = pd.DataFrame(X, columns=feature_names)
    dummy_model = lgb.LGBMClassifier(n_estimators=10, random_state=42)
    dummy_model.fit(df_train, y)

    model_name = "LightGBMFraudDetector"
    with mlflow.start_run() as run:
        mlflow.lightgbm.log_model(
            dummy_model, "lightgbm_fraud_model", registered_model_name=model_name
        )
        # Transition this model to Production stage
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name, version=1, stage="Production"
        )
        run_id_v1 = run.info.run_id

    # 2. Initialize FeatureImpactMonitor
    monitor = FeatureImpactMonitor(config_directory, "dev", model_name)
    monitor.data_lake_client = MockDataLakeClientFI(
        monitor.config["data_lake_config"]
    )  # Inject mock client
    monitor.monitor_interval_hours = 0.01  # Short interval for demo

    print("\n--- First run of Feature Impact Monitor (no history) ---")
    monitor.monitor_feature_impact()
    print(f"Last logged importances: {monitor.last_logged_importances}")

    # 3. Simulate retraining and significant feature change
    print("\n--- Simulate model retraining with feature shift ---")
    # Change importance of feature_0
    dummy_model_v2 = lgb.LGBMClassifier(
        n_estimators=10, random_state=43
    )  # Slightly different
    X_v2 = X.copy()
    X_v2[:, 0] = X_v2[:, 0] * 5  # Make feature_0 much more impactful
    dummy_model_v2.fit(X_v2, y)
    dummy_model_v2.feature_name_ = feature_names  # Ensure feature names are set

    with mlflow.start_run() as run_v2:
        mlflow.lightgbm.log_model(
            dummy_model_v2, "lightgbm_fraud_model", registered_model_name=model_name
        )
        client.transition_model_version_stage(
            name=model_name, version=2, stage="Production"
        )
        run_id_v2 = run_v2.info.run_id

    # Run monitor again, should detect change
    print("\n--- Second run of Feature Impact Monitor (with shift) ---")
    monitor.monitor_feature_impact()
    print(f"Last logged importances after shift: {monitor.last_logged_importances}")

    # Clean up dummy files
    if mlruns_dir.exists():
        import shutil

        shutil.rmtree(mlruns_dir)
    if data_vault_directory.exists():
        import shutil

        shutil.rmtree(data_vault_directory / monitor.feature_importance_log_prefix)
        # shutil.rmtree(data_vault_directory) # Be careful not to delete real data
    (config_directory / "environments" / "dev.yaml").unlink(missing_ok=True)
    print("\nDemo cleanup complete.")
