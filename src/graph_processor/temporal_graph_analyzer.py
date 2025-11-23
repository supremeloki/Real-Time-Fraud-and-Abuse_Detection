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
