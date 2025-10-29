import pandas as pd
import numpy as np
import logging
from collections import deque
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)

class StreamAnomalyDetector:
    def __init__(self, window_size: int = 1000, z_score_threshold: float = 3.0):
        self.window_size = window_size
        self.z_score_threshold = z_score_threshold
        self.feature_history = {"fare_amount": deque(maxlen=window_size),"distance_km": deque(maxlen=window_size),"duration_min": deque(maxlen=window_size)}
        logger.info(f"StreamAnomalyDetector initialized with window_size={window_size}, z_score_threshold={z_score_threshold}")

    def _update_history(self, event: Dict[str, Any]):
        for feature in self.feature_history.keys():
            if feature in event:
                self.feature_history[feature].append(event[feature])

    def detect_anomalies(self, event: Dict[str, Any]) -> Tuple[bool, Dict[str, float]]:
        self._update_history(event)
        
        anomaly_scores = {}
        is_overall_anomaly = False

        for feature, history_deque in self.feature_history.items():
            if len(history_deque) < 10: # Minimum data points to calculate meaningful std dev
                anomaly_scores[feature] = 0.0
                continue
            
            data = np.array(list(history_deque))
            mean_val = np.mean(data)
            std_dev = np.std(data)

            if std_dev == 0:
                anomaly_scores[feature] = 0.0
                continue

            current_value = event.get(feature, mean_val)
            z_score = abs((current_value - mean_val) / std_dev)
            
            anomaly_scores[feature] = z_score
            if z_score >= self.z_score_threshold:
                is_overall_anomaly = True
                logger.warning(f"Anomaly detected for '{feature}': value={current_value}, z-score={z_score:.2f}")

        return is_overall_anomaly, anomaly_scores