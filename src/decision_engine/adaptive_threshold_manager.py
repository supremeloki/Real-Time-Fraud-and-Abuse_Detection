# src/decision_engine/adaptive_threshold_manager.py

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

# Add project root to Python path for imports
import sys
from pathlib import Path

current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data_access.redis_cache_manager import (
    RedisCacheManager,
)  # Assuming RedisCacheManager exists
from src.utils.common_helpers import setup_logging

logger = logging.getLogger(__name__)


class AdaptiveThresholdManager:
    """
    Dynamically adjusts fraud detection thresholds based on real-time performance metrics
    and operational feedback.
    """

    def __init__(
        self,
        redis_config: Dict[str, Any],
        initial_thresholds: Dict[str, float],
        adjustment_factor: float = 0.01,
        min_threshold: float = 0.1,
        max_threshold: float = 0.9,
    ):

        self.redis_client = RedisCacheManager(redis_config)
        self.current_thresholds = initial_thresholds.copy()
        self.adjustment_factor = adjustment_factor
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

        self.performance_metric_key = "threshold_metrics:last_day"
        self.last_adjustment_timestamp: Optional[datetime] = None

        self._load_current_thresholds()
        logger.info(
            f"AdaptiveThresholdManager initialized. Current thresholds: {self.current_thresholds}"
        )

    def _load_current_thresholds(self):
        """Loads the latest thresholds from Redis, or uses initial if not found."""
        stored_thresholds = self.redis_client.get_value("adaptive_thresholds:current")
        if stored_thresholds:
            self.current_thresholds.update(stored_thresholds)
            logger.info("Loaded current thresholds from Redis.")
        else:
            self._save_current_thresholds()  # Save initial thresholds if none existed

    def _save_current_thresholds(self):
        """Saves the current thresholds to Redis."""
        self.redis_client.set_value(
            "adaptive_thresholds:current", self.current_thresholds
        )
        logger.info("Saved current thresholds to Redis.")

    def update_performance_metrics(self, new_metrics: Dict[str, Any]):
        """
        Updates the internal performance metrics based on recent evaluation.
        Expected metrics: 'precision', 'recall', 'false_positive_rate', 'false_negative_rate'.
        """
        self.redis_client.set_value(
            self.performance_metric_key,
            new_metrics,
            ttl_seconds=timedelta(days=1).total_seconds(),
        )
        logger.info(f"Updated performance metrics in Redis: {new_metrics}")

    def _get_recent_performance(self) -> Dict[str, Any]:
        """Retrieves recent performance metrics from Redis."""
        metrics = self.redis_client.get_value(self.performance_metric_key)
        return metrics if metrics else {}

    def adjust_thresholds(
        self,
        target_metric: str = "precision",
        target_value: float = 0.85,
        action_type: str = "manual_review_queue_score",
    ):
        """
        Adjusts a specific threshold (e.g., 'manual_review_queue_score') based on
        a target performance metric and its desired value.
        """
        if datetime.now() - (
            self.last_adjustment_timestamp or datetime.min
        ) < timedelta(hours=1):
            logger.info("Too soon to adjust thresholds. Skipping.")
            return

        recent_metrics = self._get_recent_performance()
        if not recent_metrics or target_metric not in recent_metrics:
            logger.warning(
                f"Insufficient recent performance metrics to adjust thresholds for {action_type}."
            )
            return

        current_metric_value = recent_metrics[target_metric]
        current_threshold = self.current_thresholds.get(action_type)

        if current_threshold is None:
            logger.warning(
                f"Threshold '{action_type}' not found in current thresholds."
            )
            return

        old_threshold = current_threshold
        if current_metric_value < target_value:
            # If performance is below target, decrease threshold to be more sensitive (catch more fraud)
            new_threshold = current_threshold - self.adjustment_factor
            logger.info(
                f"Decreasing '{action_type}' threshold from {old_threshold:.2f} due to low {target_metric} ({current_metric_value:.2f} < {target_value:.2f})."
            )
        elif (
            current_metric_value > target_value + self.adjustment_factor
        ):  # Add buffer to avoid constant minor changes
            # If performance is significantly above target, increase threshold to be less sensitive (reduce false positives)
            new_threshold = current_threshold + self.adjustment_factor
            logger.info(
                f"Increasing '{action_type}' threshold from {old_threshold:.2f} due to high {target_metric} ({current_metric_value:.2f} > {target_value:.2f})."
            )
        else:
            logger.info(
                f"Performance for {target_metric} is stable around target. No adjustment needed for '{action_type}'."
            )
            return

        # Clamp threshold within min/max bounds
        new_threshold = max(self.min_threshold, min(self.max_threshold, new_threshold))

        if new_threshold != old_threshold:
            self.current_thresholds[action_type] = new_threshold
            self._save_current_thresholds()
            self.last_adjustment_timestamp = datetime.now()
            logger.info(f"Threshold '{action_type}' adjusted to {new_threshold:.2f}.")
        else:
            logger.info(
                f"Threshold '{action_type}' remained {new_threshold:.2f} after clamping."
            )

    def get_current_thresholds(self) -> Dict[str, float]:
        """Returns the currently active fraud detection thresholds."""
        return self.current_thresholds


if __name__ == "__main__":
    print("AdaptiveThresholdManager - Module loaded successfully")
    print(
        "Note: Full execution requires Redis running. This module is designed to run within the main fraud detection system"
    )
    exit(0)  # Exit gracefully since this is a library module
