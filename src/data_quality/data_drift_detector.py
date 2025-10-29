import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from scipy.stats import ks_2samp, chi2_contingency

# Add project root to Python path for imports
import sys
from pathlib import Path

current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.common_helpers import setup_logging
from src.data_access.data_lake_client import DataLakeClient

logger = logging.getLogger(__name__)


class DataDriftDetector:
    def __init__(self, config: Dict[str, Any]):
        self.data_lake_client = DataLakeClient(config["data_lake_config"])
        self.reference_data_path = config["data_drift_config"]["reference_data_path"]
        self.current_data_log_path = config["data_drift_config"][
            "current_data_log_path"
        ]
        self.monitoring_interval_hours = config["data_drift_config"][
            "monitoring_interval_hours"
        ]
        self.drift_threshold_ks = config["data_drift_config"].get(
            "drift_threshold_ks", 0.05
        )  # p-value for KS test
        self.drift_threshold_chi2 = config["data_drift_config"].get(
            "drift_threshold_chi2", 0.05
        )  # p-value for Chi-squared test
        self.min_samples_for_drift = config["data_drift_config"].get(
            "min_samples_for_drift", 100
        )
        self.last_monitor_timestamp: Optional[datetime] = None
        self.reference_data: Optional[pd.DataFrame] = None
        logger.info(
            f"DataDriftDetector initialized. Monitoring interval: {self.monitoring_interval_hours} hours."
        )

    def _load_reference_data(self) -> Optional[pd.DataFrame]:
        if self.reference_data is not None:
            return self.reference_data

        # Assume reference data is a single CSV in the data lake
        path_obj = Path(self.reference_data_path)
        df = self.data_lake_client.load_dataframe_from_s3(
            path_obj.parent.name, path_obj.name, file_format="csv"
        )
        if df is not None:
            self.reference_data = df
            logger.info(
                f"Loaded reference data with {len(df)} samples from {self.reference_data_path}."
            )
        return self.reference_data

    def _load_current_data(self) -> Optional[pd.DataFrame]:
        # Assume current data is collected hourly/daily into a specific log path
        # For simplicity, load all data within the last monitoring interval
        now = datetime.now()
        start_time = now - timedelta(hours=self.monitoring_interval_hours)

        # In a real system, iterate over daily/hourly partitions
        # For demo, assume a single file containing recent data
        path_obj = Path(self.current_data_log_path)
        df = self.data_lake_client.load_dataframe_from_s3(
            path_obj.parent.name, path_obj.name, file_format="csv"
        )

        if df is not None and "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df[df["timestamp"] >= start_time]
            logger.info(
                f"Loaded {len(df)} current data samples from {self.current_data_log_path} within the monitoring window."
            )
        elif df is not None:
            logger.warning(
                "Current data log does not have a 'timestamp' column for filtering."
            )

        return df

    def detect_drift(self) -> Dict[str, Any]:
        """Detects data drift between reference and current data."""
        if not self.last_monitor_timestamp:
            self.last_monitor_timestamp = datetime.now() - timedelta(
                hours=self.monitoring_interval_hours + 1
            )  # Force first run

        if (
            datetime.now() - self.last_monitor_timestamp
        ).total_seconds() / 3600 < self.monitoring_interval_hours:
            logger.info("Data drift detection not due yet.")
            return {}

        logger.info("Starting data drift detection.")

        ref_df = self._load_reference_data()
        current_df = self._load_current_data()

        if ref_df is None or current_df is None or ref_df.empty or current_df.empty:
            logger.warning(
                "Insufficient data for drift detection. Reference or current data is empty."
            )
            return {}

        if len(current_df) < self.min_samples_for_drift:
            logger.warning(
                f"Not enough current data samples ({len(current_df)}) for robust drift detection. Min required: {self.min_samples_for_drift}."
            )
            return {}

        drift_results: Dict[str, Any] = {}
        common_features = list(
            set(ref_df.columns) & set(current_df.columns) - {"timestamp"}
        )  # Exclude timestamp for direct comparison

        for feature in common_features:
            if feature not in ref_df.columns or feature not in current_df.columns:
                continue

            ref_series = ref_df[feature].dropna()
            current_series = current_df[feature].dropna()

            if ref_series.empty or current_series.empty:
                logger.debug(
                    f"Skipping drift check for feature '{feature}' due to empty series."
                )
                continue

            try:
                if pd.api.types.is_numeric_dtype(
                    ref_series
                ) and pd.api.types.is_numeric_dtype(current_series):
                    # Kolmogorov-Smirnov test for numerical features
                    if len(ref_series) < 2 or len(current_series) < 2:
                        logger.debug(
                            f"Skipping KS test for {feature} due to too few samples."
                        )
                        continue
                    statistic, p_value = ks_2samp(ref_series, current_series)
                    if p_value < self.drift_threshold_ks:
                        drift_results[feature] = {
                            "type": "Numerical_Drift",
                            "ks_statistic": float(statistic),
                            "p_value": float(p_value),
                            "alert": True,
                            "message": f"Significant drift detected (p={p_value:.4f}) in numerical feature '{feature}'.",
                        }
                        logger.warning(drift_results[feature]["message"])
                elif pd.api.types.is_categorical_dtype(
                    ref_series
                ) or pd.api.types.is_object_dtype(ref_series):
                    # Chi-squared test for categorical features
                    ref_counts = ref_series.value_counts()
                    current_counts = current_series.value_counts()

                    combined_counts = pd.DataFrame(
                        {"reference": ref_counts, "current": current_counts}
                    ).fillna(0)

                    if combined_counts.shape[0] < 2 or combined_counts.sum().sum() == 0:
                        logger.debug(
                            f"Skipping Chi2 test for {feature} due to too few categories or total counts."
                        )
                        continue

                    # Chi-squared needs expected counts to be > 0. Handle sparse categories.
                    # Simple approach: filter out categories with very low combined counts.
                    combined_counts = combined_counts[combined_counts.sum(axis=1) > 0]
                    if combined_counts.shape[0] < 2:
                        logger.debug(
                            f"Skipping Chi2 test for {feature} after filtering sparse categories."
                        )
                        continue

                    chi2_stat, p_value, _, _ = chi2_contingency(combined_counts)
                    if p_value < self.drift_threshold_chi2:
                        drift_results[feature] = {
                            "type": "Categorical_Drift",
                            "chi2_statistic": float(chi2_stat),
                            "p_value": float(p_value),
                            "alert": True,
                            "message": f"Significant drift detected (p={p_value:.4f}) in categorical feature '{feature}'.",
                        }
                        logger.warning(drift_results[feature]["message"])
            except Exception as e:
                logger.error(
                    f"Error during drift detection for feature '{feature}': {e}",
                    exc_info=True,
                )

        if drift_results:
            logger.info(
                f"Data drift analysis complete. Detected drift in {len(drift_results)} features."
            )
        else:
            logger.info("No significant data drift detected.")

        self.last_monitor_timestamp = datetime.now()
        return drift_results


if __name__ == "__main__":
    print("DataDriftDetector - Module loaded successfully")
    print(
        "Note: Full execution requires S3 access and configuration. This module is designed to run within the main fraud detection system"
    )
    exit(0)  # Exit gracefully since this is a library module
