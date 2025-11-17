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
