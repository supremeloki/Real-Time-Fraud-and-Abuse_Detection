import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List
from src.data_access.redis_cache_manager import (
    RedisCacheManager,
)  # Assuming RedisCacheManager exists

logger = logging.getLogger(__name__)


class UserBehavioralProfiler:
    def __init__(
        self,
        redis_config: Dict[str, Any],
        profile_windows_days: List[int] = [7, 30, 90],
    ):
        self.redis_client = RedisCacheManager(redis_config)
        self.profile_windows_days = sorted(profile_windows_days)
        self.max_history_days = (
            max(profile_windows_days) + 1
        )  # Keep history slightly longer than max window
        logger.info(
            f"UserBehavioralProfiler initialized with profile windows: {profile_windows_days} days."
        )

    def _get_user_profile_key(self, user_id: str) -> str:
        return f"user_profile:{user_id}:events"

    def update_user_profile(self, event: Dict[str, Any]):
        user_id = event.get("user_id")
        if not user_id:
            logger.warning("Event without 'user_id' cannot update user profile.")
            return

        event_timestamp_str = event.get("event_timestamp")
        if not event_timestamp_str:
            logger.warning(
                f"Event {event.get('event_id')} has no timestamp. Skipping profile update."
            )
            return

        try:
            event_timestamp = datetime.fromisoformat(event_timestamp_str)
        except ValueError:
            logger.error(
                f"Invalid timestamp format for event {event.get('event_id')}: {event_timestamp_str}"
            )
            return

        profile_key = self._get_user_profile_key(user_id)

        # Store raw event data with timestamp in a sorted set (ZADD) by timestamp
        # Use a high TTL (e.g., 90 days + buffer) for the event history
        event_score = event_timestamp.timestamp()
        event_member = self.redis_client.set_value(
            f"event:{event.get('event_id')}",
            event,
            ttl_seconds=int(timedelta(days=self.max_history_days).total_seconds()),
        )

        # Store a reference in the user's event stream in Redis sorted set
        self.redis_client.redis_client.zadd(
            profile_key, {event.get("event_id"): event_score}
        )

        # Remove old events from the sorted set to keep it manageable
        cutoff_timestamp = (
            event_timestamp - timedelta(days=self.max_history_days)
        ).timestamp()
        self.redis_client.redis_client.zremrangebyscore(
            profile_key, "-inf", cutoff_timestamp
        )

        logger.debug(
            f"User profile for {user_id} updated with event {event.get('event_id')}."
        )

    def get_user_behavioral_features(
        self, user_id: str, current_timestamp: datetime
    ) -> Dict[str, Any]:
        features = {}
        profile_key = self._get_user_profile_key(user_id)

        # Fetch event IDs for all relevant time windows
        all_event_ids_in_max_window = self.redis_client.redis_client.zrangebyscore(
            profile_key,
            (current_timestamp - timedelta(days=self.max_history_days)).timestamp(),
            current_timestamp.timestamp(),
        )

        if not all_event_ids_in_max_window:
            logger.debug(f"No recent events found for user {user_id}.")
            return features

        # Retrieve full event data for these IDs
        pipeline = self.redis_client.redis_client.pipeline()
        for event_id in all_event_ids_in_max_window:
            pipeline.get(f"event:{event_id}")
        raw_events_data = pipeline.execute()

        events = []
        for event_json in raw_events_data:
            if event_json:
                try:
                    event_data = json.loads(event_json)
                    event_data["event_timestamp"] = datetime.fromisoformat(
                        event_data["event_timestamp"]
                    )
                    events.append(event_data)
                except (json.JSONDecodeError, ValueError) as e:
                    logger.error(f"Error decoding or parsing event from Redis: {e}")

        if not events:
            return features

        all_events_df = pd.DataFrame(events)

        for window_days in self.profile_windows_days:
            window_start_time = current_timestamp - timedelta(days=window_days)
            window_df = all_events_df[
                all_events_df["event_timestamp"] >= window_start_time
            ]

            if not window_df.empty:
                prefix = f"user_behavior_{window_days}d"
                features[f"{prefix}_total_rides"] = len(window_df)
                features[f"{prefix}_avg_fare"] = window_df["fare_amount"].mean()
                features[f"{prefix}_unique_drivers"] = window_df["driver_id"].nunique()
                features[f"{prefix}_avg_distance"] = window_df["distance_km"].mean()
                features[f"{prefix}_promo_usage_rate"] = (
                    window_df["promo_code_used"].notna().mean()
                )
                features[f"{prefix}_cancellation_rate"] = (
                    window_df["event_type"] == "ride_cancelled"
                ).mean()
                features[f"{prefix}_recent_ride_interval_mean"] = (
                    window_df["event_timestamp"].diff().mean().total_seconds() / 60
                    if len(window_df) > 1
                    else 0
                )

            else:
                prefix = f"user_behavior_{window_days}d"
                features[f"{prefix}_total_rides"] = 0
                features[f"{prefix}_avg_fare"] = 0.0
                features[f"{prefix}_unique_drivers"] = 0
                features[f"{prefix}_avg_distance"] = 0.0
                features[f"{prefix}_promo_usage_rate"] = 0.0
                features[f"{prefix}_cancellation_rate"] = 0.0
                features[f"{prefix}_recent_ride_interval_mean"] = 0.0

        logger.debug(
            f"Generated {len(features)} behavioral features for user {user_id}."
        )
        return features


if __name__ == "__main__":
    from src.utils.common_helpers import setup_logging
    import json

    setup_logging("UserBehavioralProfilerDemo", level="INFO")

    # Assuming a local Redis instance for demonstration
    redis_config_for_demo = {
        "redis_host": "localhost",
        "redis_port": 6379,
        "redis_db": 1,  # Use a different DB to not clash with other demos
        "default_ttl_seconds": 2592000,  # 30 days
    }

    profiler = UserBehavioralProfiler(
        redis_config_for_demo, profile_windows_days=[1, 7]
    )

    try:
        profiler.redis_client.redis_client.ping()
        print("Connected to Redis for demo.")
    except Exception as e:
        print(
            f"Could not connect to Redis: {e}. Please ensure Redis is running on localhost:6379."
        )
        exit()

    test_user_id = "user_profile_test_1"
    test_driver_id = "driver_profile_test_A"

    # Simulate events over a few days
    current_time = datetime.now()
    events_to_simulate = []

    for i in range(10):  # Events within 1 day window
        event_time = current_time - timedelta(hours=i * 2)
        events_to_simulate.append(
            {
                "event_id": f"e{i:03d}",
                "event_timestamp": event_time.isoformat(),
                "event_type": "ride_completed",
                "user_id": test_user_id,
                "driver_id": f"{test_driver_id}_{i%3}",
                "fare_amount": np.random.uniform(40000, 80000),
                "distance_km": np.random.uniform(5, 15),
                "duration_min": np.random.uniform(10, 30),
                "promo_code_used": "PROMO_X" if i % 5 == 0 else None,
            }
        )

    # Event older than 1 day but within 7 days
    event_time_old_1 = current_time - timedelta(days=2)
    events_to_simulate.append(
        {
            "event_id": f"e100",
            "event_timestamp": event_time_old_1.isoformat(),
            "event_type": "ride_completed",
            "user_id": test_user_id,
            "driver_id": f"{test_driver_id}_old",
            "fare_amount": np.random.uniform(40000, 80000),
            "distance_km": np.random.uniform(5, 15),
            "duration_min": np.random.uniform(10, 30),
            "promo_code_used": None,
        }
    )

    # Event even older, should fall out of 7-day window if max_history_days is 7
    event_time_very_old = current_time - timedelta(days=8)
    events_to_simulate.append(
        {
            "event_id": f"e200",
            "event_timestamp": event_time_very_old.isoformat(),
            "event_type": "ride_completed",
            "user_id": test_user_id,
            "driver_id": f"{test_driver_id}_very_old",
            "fare_amount": np.random.uniform(40000, 80000),
            "distance_km": np.random.uniform(5, 15),
            "duration_min": np.random.uniform(10, 30),
            "promo_code_used": None,
        }
    )

    print("--- Updating User Profile with Simulated Events ---")
    for event in events_to_simulate:
        profiler.update_user_profile(event)

    print("\n--- Generating Behavioral Features for User ---")
    user_features = profiler.get_user_behavioral_features(test_user_id, current_time)
    print(json.dumps(user_features, indent=2))

    # Clean up for demo
    profiler.redis_client.redis_client.delete(
        profiler._get_user_profile_key(test_user_id)
    )
    for event in events_to_simulate:
        profiler.redis_client.delete_key(f"event:{event['event_id']}")
