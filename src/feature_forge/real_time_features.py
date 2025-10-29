import redis
import json
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from src.utils.common_helpers import (
    load_config,
    setup_logging,
    serialize_json_with_datetime,
    deserialize_json_with_datetime,
)

logger = setup_logging(__name__)


class RealTimeFeatureStore:
    def __init__(self, config_path: Path, env: str):
        self.config = load_config(config_path, env)
        self.logger = setup_logging(
            "FeatureStore", self.config["environment"]["log_level"]
        )

        redis_host = self.config["environment"]["redis_host"]
        redis_port = self.config["environment"]["redis_port"]
        self.redis_client = redis.StrictRedis(host=redis_host, port=redis_port, db=0)
        self.logger.info(f"Connected to Redis at {redis_host}:{redis_port}")

        self.time_window_seconds_short = 300  # 5 minutes
        self.time_window_seconds_medium = 1800  # 30 minutes
        self.time_window_seconds_long = 3600  # 1 hour

    def store_event(self, event: dict):
        event_id = event.get("event_id")
        if not event_id:
            self.logger.warning("Event without 'event_id' cannot be stored.")
            return

        key = f"event:{event_id}"
        self.redis_client.setex(
            key, timedelta(hours=24), serialize_json_with_datetime(event)
        )
        self.logger.debug(f"Event {event_id} stored in Redis.")

    def update_user_features(
        self,
        user_id: str,
        event_time: datetime,
        event_type: str,
        fare_amount: float = 0,
        distance_km: float = 0,
    ):
        user_key = f"user_features:{user_id}"

        pipeline = self.redis_client.pipeline()

        pipeline.zremrangebyscore(
            f"{user_key}:ride_timestamps",
            "-inf",
            (event_time - timedelta(seconds=self.time_window_long)).timestamp(),
        )
        pipeline.zadd(
            f"{user_key}:ride_timestamps",
            {event_time.timestamp(): event_time.timestamp()},
        )

        pipeline.incr(f"{user_key}:total_events")
        pipeline.incrbyfloat(f"{user_key}:total_fare", fare_amount)
        pipeline.incrbyfloat(f"{user_key}:total_distance", distance_km)

        if event_type == "ride_completed":
            pipeline.incr(f"{user_key}:rides_completed_count")
            pipeline.incr(f"{user_key}:rides_completed_short_window", 1)
            pipeline.incr(f"{user_key}:rides_completed_medium_window", 1)
            pipeline.incr(f"{user_key}:rides_completed_long_window", 1)

            pipeline.expire(
                f"{user_key}:rides_completed_short_window",
                self.time_window_seconds_short,
            )
            pipeline.expire(
                f"{user_key}:rides_completed_medium_window",
                self.time_window_seconds_medium,
            )
            pipeline.expire(
                f"{user_key}:rides_completed_long_window", self.time_window_seconds_long
            )

        pipeline.execute()
        self.logger.debug(f"User {user_id} features updated in Redis.")

    def update_driver_features(
        self,
        driver_id: str,
        event_time: datetime,
        event_type: str,
        fare_amount: float = 0,
        distance_km: float = 0,
    ):
        driver_key = f"driver_features:{driver_id}"

        pipeline = self.redis_client.pipeline()

        pipeline.zremrangebyscore(
            f"{driver_key}:ride_timestamps",
            "-inf",
            (event_time - timedelta(seconds=self.time_window_long)).timestamp(),
        )
        pipeline.zadd(
            f"{driver_key}:ride_timestamps",
            {event_time.timestamp(): event_time.timestamp()},
        )

        pipeline.incr(f"{driver_key}:total_events")
        pipeline.incrbyfloat(f"{driver_key}:total_fare", fare_amount)
        pipeline.incrbyfloat(f"{driver_key}:total_distance", distance_km)

        if event_type == "ride_completed":
            pipeline.incr(f"{driver_key}:rides_completed_count")
            pipeline.incr(f"{driver_key}:rides_completed_short_window", 1)
            pipeline.incr(f"{driver_key}:rides_completed_medium_window", 1)
            pipeline.incr(f"{driver_key}:rides_completed_long_window", 1)

            pipeline.expire(
                f"{driver_key}:rides_completed_short_window",
                self.time_window_seconds_short,
            )
            pipeline.expire(
                f"{driver_key}:rides_completed_medium_window",
                self.time_window_seconds_medium,
            )
            pipeline.expire(
                f"{driver_key}:rides_completed_long_window",
                self.time_window_seconds_long,
            )

        pipeline.execute()
        self.logger.debug(f"Driver {driver_id} features updated in Redis.")

    def get_user_features(self, user_id: str) -> dict:
        user_key = f"user_features:{user_id}"

        num_rides_short = (
            self.redis_client.get(f"{user_key}:rides_completed_short_window") or 0
        )
        num_rides_medium = (
            self.redis_client.get(f"{user_key}:rides_completed_medium_window") or 0
        )
        num_rides_long = (
            self.redis_client.get(f"{user_key}:rides_completed_long_window") or 0
        )

        features = {
            "user_total_events": int(
                self.redis_client.get(f"{user_key}:total_events") or 0
            ),
            "user_total_fare": float(
                self.redis_client.get(f"{user_key}:total_fare") or 0
            ),
            "user_total_distance": float(
                self.redis_client.get(f"{user_key}:total_distance") or 0
            ),
            "user_rides_completed_5min": int(num_rides_short),
            "user_rides_completed_30min": int(num_rides_medium),
            "user_rides_completed_1hr": int(num_rides_long),
            "user_ride_frequency_1hr": self.redis_client.zcard(
                f"{user_key}:ride_timestamps"
            ),
        }
        self.logger.debug(f"Retrieved features for user {user_id}: {features}")
        return features

    def get_driver_features(self, driver_id: str) -> dict:
        driver_key = f"driver_features:{driver_id}"

        num_rides_short = (
            self.redis_client.get(f"{driver_key}:rides_completed_short_window") or 0
        )
        num_rides_medium = (
            self.redis_client.get(f"{driver_key}:rides_completed_medium_window") or 0
        )
        num_rides_long = (
            self.redis_client.get(f"{driver_key}:rides_completed_long_window") or 0
        )

        features = {
            "driver_total_events": int(
                self.redis_client.get(f"{driver_key}:total_events") or 0
            ),
            "driver_total_fare": float(
                self.redis_client.get(f"{driver_key}:total_fare") or 0
            ),
            "driver_total_distance": float(
                self.redis_client.get(f"{driver_key}:total_distance") or 0
            ),
            "driver_rides_completed_5min": int(num_rides_short),
            "driver_rides_completed_30min": int(num_rides_medium),
            "driver_rides_completed_1hr": int(num_rides_long),
            "driver_ride_frequency_1hr": self.redis_client.zcard(
                f"{driver_key}:ride_timestamps"
            ),
        }
        self.logger.debug(f"Retrieved features for driver {driver_id}: {features}")
        return features


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Real-time Feature Store for Snapp Fraud Detection"
    )
    parser.add_argument(
        "--env", type=str, default="dev", help="Environment (dev or prod)"
    )
    args = parser.parse_args()

    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    config_directory = project_root / "config"

    feature_store = RealTimeFeatureStore(config_directory, args.env)

    test_user_id = "test_user_123"
    test_driver_id = "test_driver_456"

    test_event_1 = {
        "event_id": "evt_001",
        "event_timestamp": datetime.now(),
        "event_type": "ride_requested",
        "user_id": test_user_id,
        "driver_id": test_driver_id,
        "ride_id": "ride_abc",
        "fare_amount": 0,
        "distance_km": 0,
    }
    test_event_2 = {
        "event_id": "evt_002",
        "event_timestamp": datetime.now() + timedelta(seconds=10),
        "event_type": "ride_completed",
        "user_id": test_user_id,
        "driver_id": test_driver_id,
        "ride_id": "ride_abc",
        "fare_amount": 50000,
        "distance_km": 5.2,
    }

    feature_store.store_event(test_event_1)
    feature_store.update_user_features(
        test_user_id, test_event_1["event_timestamp"], test_event_1["event_type"]
    )
    feature_store.update_driver_features(
        test_driver_id, test_event_1["event_timestamp"], test_event_1["event_type"]
    )

    feature_store.store_event(test_event_2)
    feature_store.update_user_features(
        test_user_id,
        test_event_2["event_timestamp"],
        test_event_2["event_type"],
        test_event_2["fare_amount"],
        test_event_2["distance_km"],
    )
    feature_store.update_driver_features(
        test_driver_id,
        test_event_2["event_timestamp"],
        test_event_2["event_type"],
        test_event_2["fare_amount"],
        test_event_2["distance_km"],
    )

    user_feats = feature_store.get_user_features(test_user_id)
    driver_feats = feature_store.get_driver_features(test_driver_id)

    print(f"User features: {user_feats}")
    print(f"Driver features: {driver_feats}")
