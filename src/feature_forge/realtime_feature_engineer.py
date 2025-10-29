import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from collections import deque
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class RealtimeFeatureEngineer:
    def __init__(self, time_windows_minutes: List[int] = [5, 30, 60]):
        self.time_windows_minutes = sorted(time_windows_minutes)
        self.user_event_history: Dict[str, deque] = {}
        self.driver_event_history: Dict[str, deque] = {}
        logger.info(
            f"RealtimeFeatureEngineer initialized with time windows: {time_windows_minutes} minutes."
        )

    def _get_entity_history(self, entity_id: str, entity_type: str):
        if entity_type == "user":
            return self.user_event_history.setdefault(entity_id, deque())
        elif entity_type == "driver":
            return self.driver_event_history.setdefault(entity_id, deque())
        return deque()

    def _clean_history(
        self, history: deque, current_timestamp: datetime, window_minutes: int
    ):
        cutoff_time = current_timestamp - timedelta(minutes=window_minutes)
        while history and history[0]["event_timestamp"] < cutoff_time:
            history.popleft()

    def generate_features(self, event: Dict[str, Any]) -> Dict[str, Any]:
        processed_features = {}
        event_timestamp = datetime.fromisoformat(event["event_timestamp"])

        processed_features["event_hour"] = event_timestamp.hour
        processed_features["event_day_of_week"] = event_timestamp.weekday()
        processed_features["fare_per_km"] = event.get("fare_amount", 0) / (
            event.get("distance_km", 1e-6) + 1e-6
        )
        processed_features["distance_per_duration"] = event.get("distance_km", 0) / (
            event.get("duration_min", 1e-6) + 1e-6
        )

        user_id = event.get("user_id")
        driver_id = event.get("driver_id")

        if user_id:
            user_history = self._get_entity_history(user_id, "user")
            user_history.append({"event_timestamp": event_timestamp, **event})

            for window in self.time_windows_minutes:
                self._clean_history(user_history, event_timestamp, window)
                window_df = pd.DataFrame(list(user_history))

                if not window_df.empty:
                    processed_features[f"user_avg_fare_{window}min"] = window_df[
                        "fare_amount"
                    ].mean()
                    processed_features[f"user_total_rides_{window}min"] = len(window_df)
                    processed_features[f"user_max_distance_{window}min"] = window_df[
                        "distance_km"
                    ].max()
                else:
                    processed_features[f"user_avg_fare_{window}min"] = 0.0
                    processed_features[f"user_total_rides_{window}min"] = 0
                    processed_features[f"user_max_distance_{window}min"] = 0.0

        if driver_id:
            driver_history = self._get_entity_history(driver_id, "driver")
            driver_history.append({"event_timestamp": event_timestamp, **event})

            for window in self.time_windows_minutes:
                self._clean_history(driver_history, event_timestamp, window)
                window_df = pd.DataFrame(list(driver_history))

                if not window_df.empty:
                    processed_features[f"driver_avg_fare_{window}min"] = window_df[
                        "fare_amount"
                    ].mean()
                    processed_features[f"driver_total_rides_{window}min"] = len(
                        window_df
                    )
                else:
                    processed_features[f"driver_avg_fare_{window}min"] = 0.0
                    processed_features[f"driver_total_rides_{window}min"] = 0

        logger.debug(
            f"Generated {len(processed_features)} real-time features for event {event.get('event_id')}."
        )
        return processed_features
