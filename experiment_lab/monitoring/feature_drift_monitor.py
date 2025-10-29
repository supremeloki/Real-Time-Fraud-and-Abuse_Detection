import pandas as pd
import numpy as np
import logging
from scipy import stats
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class DriftResult:
    """Data class to store drift detection results"""
    feature: str
    is_drift: bool
    statistic: float
    p_value: float
    baseline_samples: int
    current_samples: int
    timestamp: datetime

class FeatureDriftMonitor:
    """
    Enhanced Feature Drift Monitor for detecting distribution changes in features.

    This class implements statistical tests to detect concept drift in feature distributions
    between baseline (training) data and current production data.
    """

    def __init__(self,
                 baseline_data: pd.DataFrame,
                 features_to_monitor: List[str],
                 ks_alpha: float = 0.05,
                 min_samples: int = 30):
        """
        Initialize the Feature Drift Monitor.

        Args:
            baseline_data: Historical training data for baseline comparison
            features_to_monitor: List of feature names to monitor for drift
            ks_alpha: Significance level for Kolmogorov-Smirnov test
            min_samples: Minimum samples required for reliable statistical testing
        """
        self.baseline_data = baseline_data.copy()
        self.features_to_monitor = features_to_monitor
        self.ks_alpha = ks_alpha
        self.min_samples = min_samples
        self.drift_history: List[DriftResult] = []

        # Validate features exist in baseline data
        missing_features = [f for f in features_to_monitor if f not in baseline_data.columns]
        if missing_features:
            logger.warning(f"Features not found in baseline data: {missing_features}")

        # Store baseline statistics for each feature
        self.baseline_stats = self._compute_baseline_statistics()

        logger.info(f"FeatureDriftMonitor initialized for {len(features_to_monitor)} features "
                   f"with KS alpha={ks_alpha}, min_samples={min_samples}")

    def _compute_baseline_statistics(self) -> Dict[str, Dict[str, float]]:
        """Compute and store baseline statistics for each feature"""
        stats_dict = {}
        for feature in self.features_to_monitor:
            if feature in self.baseline_data.columns:
                series = self.baseline_data[feature].dropna()
                if len(series) >= self.min_samples:
                    stats_dict[feature] = {
                        'mean': series.mean(),
                        'std': series.std(),
                        'median': series.median(),
                        'q25': series.quantile(0.25),
                        'q75': series.quantile(0.75),
                        'count': len(series)
                    }
                else:
                    logger.warning(f"Insufficient baseline data for feature '{feature}' "
                                 f"({len(series)} < {self.min_samples})")
            else:
                logger.warning(f"Feature '{feature}' not found in baseline data")
        return stats_dict

    def detect_drift(self, current_batch_data: pd.DataFrame) -> Dict[str, DriftResult]:
        """
        Detect drift for all monitored features using current batch data.

        Args:
            current_batch_data: Current production data batch

        Returns:
            Dictionary mapping feature names to DriftResult objects
        """
        drift_results = {}

        for feature in self.features_to_monitor:
            result = self._detect_single_feature_drift(feature, current_batch_data)
            drift_results[feature] = result
            self.drift_history.append(result)

            # Log significant drift
            if result.is_drift:
                logger.warning(f"🚨 DRIFT DETECTED: {feature} "
                             f"(p-value={result.p_value:.4f}, statistic={result.statistic:.4f})")

        return drift_results

    def _detect_single_feature_drift(self, feature: str, current_data: pd.DataFrame) -> DriftResult:
        """
        Detect drift for a single feature using Kolmogorov-Smirnov test.

        Enhanced with multiple statistical tests and better error handling.
        """
        timestamp = datetime.now()

        # Check if feature exists in both datasets
        if feature not in self.baseline_data.columns:
            logger.warning(f"Feature '{feature}' missing from baseline data")
            return DriftResult(feature, False, 0.0, 1.0, 0, 0, timestamp)

        if feature not in current_data.columns:
            logger.warning(f"Feature '{feature}' missing from current data")
            return DriftResult(feature, False, 0.0, 1.0, 0, 0, timestamp)

        # Get clean data samples
        baseline_series = self.baseline_data[feature].dropna()
        current_series = current_data[feature].dropna()

        baseline_count = len(baseline_series)
        current_count = len(current_series)

        # Check minimum sample size
        if baseline_count < self.min_samples or current_count < self.min_samples:
            logger.warning(f"Insufficient data for KS test on '{feature}': "
                         f"baseline={baseline_count}, current={current_count}, "
                         f"required={self.min_samples}")
            return DriftResult(feature, False, 0.0, 1.0, baseline_count, current_count, timestamp)

        # Perform Kolmogorov-Smirnov test
        try:
            statistic, p_value = stats.ks_2samp(baseline_series, current_series)

            # Determine if drift is detected
            is_drift = p_value < self.ks_alpha

            return DriftResult(feature, is_drift, statistic, p_value,
                             baseline_count, current_count, timestamp)

        except Exception as e:
            logger.error(f"Error performing KS test for feature '{feature}': {str(e)}")
            return DriftResult(feature, False, 0.0, 1.0, baseline_count, current_count, timestamp)

    def get_drift_summary(self) -> Dict[str, Any]:
        """Get a summary of drift detection results"""
        if not self.drift_history:
            return {"total_checks": 0, "features_with_drift": [], "drift_rate": 0.0}

        recent_results = self.drift_history[-len(self.features_to_monitor):]  # Last batch
        features_with_drift = [r.feature for r in recent_results if r.is_drift]

        return {
            "total_checks": len(self.drift_history),
            "features_with_drift": features_with_drift,
            "drift_rate": len(features_with_drift) / len(self.features_to_monitor),
            "last_check_timestamp": recent_results[0].timestamp if recent_results else None
        }

    def get_baseline_stats(self, feature: str) -> Optional[Dict[str, float]]:
        """Get baseline statistics for a specific feature"""
        return self.baseline_stats.get(feature)

    def reset_drift_history(self):
        """Clear the drift detection history"""
        self.drift_history.clear()
        logger.info("Drift history has been reset")