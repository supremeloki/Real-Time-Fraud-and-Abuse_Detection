# src/graph_processor/temporal_graph_analyzer.py

import networkx as nx
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, Any, List, Tuple, Optional
from src.data_access.redis_cache_manager import (
    RedisCacheManager,
)  # Assuming RedisCacheManager exists

logger = logging.getLogger(__name__)


class TemporalGraphAnalyzer:
    def __init__(
        self,
        redis_config: Dict[str, Any],
        time_window_hours: int = 24,
        max_history_days: int = 7,
    ):
        self.redis_client = RedisCacheManager(redis_config)
        self.time_window_hours = time_window_hours
        self.max_history_days = (
            max_history_days  # Max days to keep individual event records in Redis
        )
        self.user_driver_interactions: Dict[Tuple[str, str], List[datetime]] = (
            defaultdict(list)
        )
        self.node_event_history: Dict[str, List[Dict[str, Any]]] = defaultdict(
            list
        )  # Track events per node in memory
        logger.info(
            f"TemporalGraphAnalyzer initialized with {time_window_hours}-hour window."
        )

    def _get_node_event_key(self, node_id: str) -> str:
        return f"node_events:{node_id}"

    def update_temporal_graph_state(self, event: Dict[str, Any]):
        event_timestamp_str = event.get("event_timestamp")
        if not event_timestamp_str:
            logger.warning(
                f"Event {event.get('event_id')} has no timestamp. Skipping temporal graph update."
            )
            return

        event_timestamp = datetime.fromisoformat(event_timestamp_str)
        user_id = event.get("user_id")
        driver_id = event.get("driver_id")
        event_id = event.get("event_id")

        # Store a lightweight version of the event in Redis for each involved node
        event_data_for_redis = {
            "event_id": event_id,
            "event_timestamp": event_timestamp_str,
            "user_id": user_id,
            "driver_id": driver_id,
            "event_type": event.get("event_type", "unknown"),
            "fare_amount": event.get("fare_amount", 0.0),
            "distance_km": event.get("distance_km", 0.0),
        }

        if user_id:
            self.redis_client.redis_client.zadd(
                self._get_node_event_key(user_id),
                {json.dumps(event_data_for_redis): event_timestamp.timestamp()},
            )
            self.redis_client.redis_client.expire(
                self._get_node_event_key(user_id),
                timedelta(days=self.max_history_days).total_seconds(),
            )
        if driver_id:
            self.redis_client.redis_client.zadd(
                self._get_node_event_key(driver_id),
                {json.dumps(event_data_for_redis): event_timestamp.timestamp()},
            )
            self.redis_client.redis_client.expire(
                self._get_node_event_key(driver_id),
                timedelta(days=self.max_history_days).total_seconds(),
            )

        # Update in-memory interaction tracking for immediate correlation
        if user_id and driver_id:
            self.user_driver_interactions[(user_id, driver_id)].append(event_timestamp)
            self._clean_interactions(user_id, driver_id, event_timestamp)

        logger.debug(f"Temporal graph state updated for event {event_id}.")

    def _clean_interactions(
        self, user_id: str, driver_id: str, current_timestamp: datetime
    ):
        # Remove old interactions from the in-memory lists
        cutoff_time = current_timestamp - timedelta(hours=self.time_window_hours)
        self.user_driver_interactions[(user_id, driver_id)] = [
            ts
            for ts in self.user_driver_interactions[(user_id, driver_id)]
            if ts >= cutoff_time
        ]

    def get_recent_node_events(
        self, node_id: str, current_timestamp: datetime
    ) -> List[Dict[str, Any]]:
        """Retrieves recent events for a given node from Redis."""
        key = self._get_node_event_key(node_id)
        min_score = (
            current_timestamp - timedelta(hours=self.time_window_hours)
        ).timestamp()
        max_score = current_timestamp.timestamp()

        raw_events = self.redis_client.redis_client.zrangebyscore(
            key, min_score, max_score
        )

        parsed_events = []
        for event_json in raw_events:
            try:
                event_data = json.loads(event_json)
                event_data["event_timestamp"] = datetime.fromisoformat(
                    event_data["event_timestamp"]
                )
                parsed_events.append(event_data)
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Error parsing event from Redis for node {node_id}: {e}")
        return parsed_events

    def analyze_temporal_patterns(
        self,
        user_id: Optional[str],
        driver_id: Optional[str],
        current_timestamp: datetime,
    ) -> Dict[str, Any]:
        """
        Analyzes temporal patterns in the graph around the given user/driver.
        Returns features indicating dynamic behaviors.
        """
        temporal_features = {}

        if user_id:
            user_events = self.get_recent_node_events(user_id, current_timestamp)
            temporal_features.update(
                self._analyze_node_temporal_patterns(
                    user_id, user_events, "user", current_timestamp
                )
            )

        if driver_id:
            driver_events = self.get_recent_node_events(driver_id, current_timestamp)
            temporal_features.update(
                self._analyze_node_temporal_patterns(
                    driver_id, driver_events, "driver", current_timestamp
                )
            )

        if user_id and driver_id:
            interaction_timestamps = self.user_driver_interactions.get(
                (user_id, driver_id), []
            )
            num_recent_interactions = len(
                [
                    ts
                    for ts in interaction_timestamps
                    if ts
                    >= (current_timestamp - timedelta(hours=self.time_window_hours))
                ]
            )
            temporal_features["user_driver_recent_shared_rides"] = (
                num_recent_interactions
            )

            if num_recent_interactions > 3:  # Heuristic for rapid interaction
                temporal_features["user_driver_rapid_interaction_spike"] = True
            else:
                temporal_features["user_driver_rapid_interaction_spike"] = False

        logger.debug(
            f"Generated temporal features for user={user_id}, driver={driver_id}."
        )
        return temporal_features

    def _analyze_node_temporal_patterns(
        self,
        node_id: str,
        events: List[Dict[str, Any]],
        node_type: str,
        current_time: datetime,
    ) -> Dict[str, Any]:
        node_temporal_features = {}

        if not events:
            return node_temporal_features

        events_df = pd.DataFrame(events)
        events_df = events_df.sort_values(by="event_timestamp")

        prefix = f"{node_type}_temporal"

        # Rate of events
        node_temporal_features[f"{prefix}_event_rate_per_hour"] = (
            len(events_df) / self.time_window_hours
        )

        # Min/Max/Avg interval between events
        if len(events_df) > 1:
            intervals = (
                events_df["event_timestamp"].diff().dropna().dt.total_seconds() / 60
            )  # in minutes
            node_temporal_features[f"{prefix}_avg_event_interval_min"] = (
                intervals.mean()
            )
            node_temporal_features[f"{prefix}_min_event_interval_min"] = intervals.min()
        else:
            node_temporal_features[f"{prefix}_avg_event_interval_min"] = 0.0
            node_temporal_features[f"{prefix}_min_event_interval_min"] = 0.0

        # Change in average fare/distance over time (trend)
        if len(events_df) > 5:  # Need enough data for a trend
            # Calculate simple linear trend of fare_amount over event_timestamp
            try:
                x = (
                    (events_df["event_timestamp"] - events_df["event_timestamp"].min())
                    .dt.total_seconds()
                    .values.reshape(-1, 1)
                )
                y_fare = events_df["fare_amount"].values
                y_distance = events_df["distance_km"].values

                from sklearn.linear_model import LinearRegression

                lr_fare = LinearRegression().fit(x, y_fare)
                lr_distance = LinearRegression().fit(x, y_distance)

                node_temporal_features[f"{prefix}_fare_trend_slope"] = lr_fare.coef_[0]
                node_temporal_features[f"{prefix}_distance_trend_slope"] = (
                    lr_distance.coef_[0]
                )
            except Exception as e:
                logger.warning(f"Could not calculate trend for {node_id}: {e}")
                node_temporal_features[f"{prefix}_fare_trend_slope"] = 0.0
                node_temporal_features[f"{prefix}_distance_trend_slope"] = 0.0
        else:
            node_temporal_features[f"{prefix}_fare_trend_slope"] = 0.0
            node_temporal_features[f"{prefix}_distance_trend_slope"] = 0.0

        return node_temporal_features


if __name__ == "__main__":
    from src.utils.common_helpers import setup_logging
    import json

    setup_logging("TemporalGraphAnalyzerDemo", level="INFO")

    redis_config_for_demo = {
        "redis_host": "localhost",
        "redis_port": 6379,
        "redis_db": 3,
        "default_ttl_seconds": 3600,
    }

    analyzer = TemporalGraphAnalyzer(
        redis_config_for_demo, time_window_hours=1, max_history_days=2
    )

    try:
        analyzer.redis_client.redis_client.ping()
        print("Connected to Redis for demo.")
    except Exception as e:
        print(
            f"Could not connect to Redis: {e}. Please ensure Redis is running on localhost:6379."
        )
        exit()

    test_user_id = "user_temp_graph_1"
    test_driver_id = "driver_temp_graph_A"
    test_driver_id_2 = "driver_temp_graph_B"

    current_time = datetime.now()

    print("--- Simulating Events to Update Temporal Graph State ---")

    # 1. Normal events for user and driver
    for i in range(3):
        event = {
            "event_id": f"n_e{i}",
            "event_timestamp": (current_time - timedelta(minutes=i * 10)).isoformat(),
            "event_type": "ride_completed",
            "user_id": test_user_id,
            "driver_id": test_driver_id,
            "fare_amount": 50000 + i * 1000,
            "distance_km": 10 + i,
            "duration_min": 20 + i,
        }
        analyzer.update_temporal_graph_state(event)

    # 2. Rapid interaction spike for user and a different driver
    for i in range(4):
        event = {
            "event_id": f"r_e{i}",
            "event_timestamp": (current_time - timedelta(minutes=1 + i)).isoformat(),
            "event_type": "ride_completed",
            "user_id": test_user_id,
            "driver_id": test_driver_id_2,
            "fare_amount": 20000 + i * 500,
            "distance_km": 2 + i * 0.5,
            "duration_min": 5 + i,
        }
        analyzer.update_temporal_graph_state(event)

    print("\n--- Analyzing Temporal Patterns for User and Driver ---")

    # Analyze for the current moment
    temporal_features = analyzer.analyze_temporal_patterns(
        test_user_id, test_driver_id_2, current_time
    )
    print(
        f"\nTemporal features for {test_user_id} and {test_driver_id_2} (current time):\n{json.dumps(temporal_features, indent=2)}"
    )

    temporal_features_old = analyzer.analyze_temporal_patterns(
        test_user_id, test_driver_id, current_time
    )
    print(
        f"\nTemporal features for {test_user_id} and {test_driver_id} (current time):\n{json.dumps(temporal_features_old, indent=2)}"
    )

    # Simulate a later time, so some events fall out of the window
    future_time = current_time + timedelta(hours=1.5)
    print(f"\n--- Analyzing Temporal Patterns at a later time ({future_time}) ---")
    temporal_features_future = analyzer.analyze_temporal_patterns(
        test_user_id, test_driver_id_2, future_time
    )
    print(
        f"\nTemporal features for {test_user_id} and {test_driver_id_2} (future time):\n{json.dumps(temporal_features_future, indent=2)}"
    )

    # Clean up Redis data
    for node_id in [test_user_id, test_driver_id, test_driver_id_2]:
        analyzer.redis_client.redis_client.delete(analyzer._get_node_event_key(node_id))
    print(
        f"\nCleaned up data for demo in Redis DB {redis_config_for_demo['redis_db']}."
    )
