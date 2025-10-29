import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple
from collections import defaultdict
import sys
from pathlib import Path

import json

current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data_access.redis_cache_manager import (
    RedisCacheManager,
)  # Assuming RedisCacheManager exists

logger = logging.getLogger(__name__)


class CrossChannelCorrelationAnalyzer:
    def __init__(
        self,
        redis_config: Dict[str, Any],
        correlation_window_minutes: int = 30,
        max_event_history: int = 1000,
    ):
        self.redis_client = RedisCacheManager(redis_config)
        self.correlation_window_minutes = correlation_window_minutes
        self.max_event_history = max_event_history
        self.entity_event_stream: Dict[str, List[Dict[str, Any]]] = defaultdict(
            list
        )  # {entity_id: list of recent events}
        logger.info(
            f"CrossChannelCorrelationAnalyzer initialized with window: {correlation_window_minutes}min."
        )

    def _clean_event_history(self, entity_id: str, current_timestamp: datetime):
        # Remove events older than the maximum correlation window
        cutoff_time = current_timestamp - timedelta(
            minutes=self.correlation_window_minutes
        )
        self.entity_event_stream[entity_id] = [
            event
            for event in self.entity_event_stream[entity_id]
            if datetime.fromisoformat(event["event_timestamp"]) >= cutoff_time
        ]
        # Keep history size bounded
        if len(self.entity_event_stream[entity_id]) > self.max_event_history:
            self.entity_event_stream[entity_id] = sorted(
                self.entity_event_stream[entity_id],
                key=lambda x: datetime.fromisoformat(x["event_timestamp"]),
            )[len(self.entity_event_stream[entity_id]) - self.max_event_history :]

    def analyze_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        analysis_results = {}
        event_timestamp = datetime.fromisoformat(event["event_timestamp"])

        user_id = event.get("user_id")
        driver_id = event.get("driver_id")

        # Update event history for user and driver
        if user_id:
            self.entity_event_stream[user_id].append(event)
            self._clean_event_history(user_id, event_timestamp)
        if driver_id:
            self.entity_event_stream[driver_id].append(event)
            self._clean_event_history(driver_id, event_timestamp)

        # Analyze for cross-channel correlations within the window
        if user_id:
            user_recent_events = self.entity_event_stream[user_id]
            analysis_results.update(
                self._analyze_user_correlations(
                    user_id, user_recent_events, event_timestamp
                )
            )

        if driver_id:
            driver_recent_events = self.entity_event_stream[driver_id]
            analysis_results.update(
                self._analyze_driver_correlations(
                    driver_id, driver_recent_events, event_timestamp
                )
            )

        # Specific patterns involving both user and driver in current event
        if user_id and driver_id:
            analysis_results.update(
                self._analyze_user_driver_pair(user_id, driver_id, event_timestamp)
            )

        logger.debug(
            f"Analyzed cross-channel correlations for event {event.get('event_id')}."
        )
        return analysis_results

    def _analyze_user_correlations(
        self, user_id: str, events: List[Dict[str, Any]], current_time: datetime
    ) -> Dict[str, Any]:
        user_correlations = {}
        window_events = [
            e
            for e in events
            if datetime.fromisoformat(e["event_timestamp"])
            >= (current_time - timedelta(minutes=self.correlation_window_minutes))
        ]

        if len(window_events) < 2:
            return user_correlations

        event_types = [e["event_type"] for e in window_events]
        unique_drivers = {e["driver_id"] for e in window_events if e.get("driver_id")}

        # High frequency of specific event types (e.g., login attempts, failed payments)
        login_attempts = sum(1 for et in event_types if "login" in et)
        if login_attempts > 5:  # Heuristic
            user_correlations["user_high_login_attempts"] = True

        # Rapid sequence of actions (e.g., login -> profile update -> ride request -> cancellation)
        sorted_events = sorted(
            window_events, key=lambda x: datetime.fromisoformat(x["event_timestamp"])
        )

        # Check for rapid login attempts from different IPs (requires IP in event history)
        distinct_ips = {
            e.get("ip_address")
            for e in sorted_events
            if e.get("ip_address") and "login" in e.get("event_type", "")
        }
        if len(distinct_ips) > 1 and login_attempts > 3:
            user_correlations["user_multi_ip_login_spike"] = True

        # Check for very short rides followed by cancellation or low fare
        short_suspicious_rides = 0
        for e in sorted_events:
            if (
                e.get("event_type") == "ride_completed"
                and e.get("distance_km", 0) < 1.0
                and e.get("fare_amount", 0) < 30000
            ):
                short_suspicious_rides += 1
        if short_suspicious_rides >= 2:
            user_correlations["user_multiple_short_low_fare_rides"] = True

        return user_correlations

    def _analyze_driver_correlations(
        self, driver_id: str, events: List[Dict[str, Any]], current_time: datetime
    ) -> Dict[str, Any]:
        driver_correlations = {}
        window_events = [
            e
            for e in events
            if datetime.fromisoformat(e["event_timestamp"])
            >= (current_time - timedelta(minutes=self.correlation_window_minutes))
        ]

        if len(window_events) < 2:
            return driver_correlations

        unique_users = {e["user_id"] for e in window_events if e.get("user_id")}

        # Driver frequently accepting very short, low-fare rides
        short_low_fare_rides = sum(
            1
            for e in window_events
            if e.get("event_type") == "ride_completed"
            and e.get("distance_km", 0) < 1.0
            and e.get("fare_amount", 0) < 30000
        )
        if short_low_fare_rides >= 2:
            driver_correlations["driver_multiple_short_low_fare_rides"] = True

        # Driver serving a very small number of unique users (potential collusion)
        if len(window_events) > 5 and len(unique_users) < (
            len(window_events) / 5
        ):  # e.g. 5 rides, 1 unique user
            driver_correlations["driver_low_unique_user_ratio"] = True

        return driver_correlations

    def _analyze_user_driver_pair(
        self, user_id: str, driver_id: str, current_time: datetime
    ) -> Dict[str, Any]:
        pair_correlations = {}

        # Check if user and driver have abnormally high number of shared rides in history
        user_events = [
            e
            for e in self.entity_event_stream[user_id]
            if datetime.fromisoformat(e["event_timestamp"])
            >= (current_time - timedelta(minutes=self.correlation_window_minutes))
        ]
        driver_events = [
            e
            for e in self.entity_event_stream[driver_id]
            if datetime.fromisoformat(e["event_timestamp"])
            >= (current_time - timedelta(minutes=self.correlation_window_minutes))
        ]

        shared_rides_count = sum(
            1 for ue in user_events if ue.get("driver_id") == driver_id
        )
        if shared_rides_count > 3 and len(user_events) > 5:  # Heuristic
            pair_correlations["user_driver_high_shared_rides"] = True

        # Co-occurrence of promo code usage
        promo_used_by_user = any(e.get("promo_code_used") for e in user_events)
        promo_used_with_driver = any(
            e.get("promo_code_used")
            for e in driver_events
            if e.get("user_id") == user_id
        )
        if promo_used_by_user and promo_used_with_driver and shared_rides_count > 0:
            pair_correlations["user_driver_promo_co_occurrence"] = True

        return pair_correlations


if __name__ == "__main__":
    print("CrossChannelCorrelationAnalyzer - Module loaded successfully")
    print("Note: Full execution requires Redis running and all dependencies installed")
    print("This module is designed to run within the main fraud detection system")

    redis_config_for_demo = {
        "redis_host": "localhost",
        "redis_port": 6379,
        "redis_db": 2,  # Use a different DB
        "default_ttl_seconds": 3600,
    }

    analyzer = CrossChannelCorrelationAnalyzer(
        redis_config_for_demo, correlation_window_minutes=60, max_event_history=50
    )

    try:
        analyzer.redis_client.redis_client.ping()
        print("Connected to Redis for demo.")
    except Exception as e:
        print(f"Could not connect to Redis: {e}. Using in-memory analysis only.")
        # Continue with demo using in-memory analysis

    test_user = "user_corr_1"
    test_driver_A = "driver_corr_A"
    test_driver_B = "driver_corr_B"

    current_time = datetime.now()
    simulated_events = []

    # Sequence 1: Normal events
    for i in range(5):
        simulated_events.append(
            {
                "event_id": f"s1_e{i}",
                "event_timestamp": (
                    current_time - timedelta(minutes=5 * i)
                ).isoformat(),
                "event_type": "ride_completed",
                "user_id": test_user,
                "driver_id": test_driver_A,
                "fare_amount": 60000,
                "distance_km": 10,
                "duration_min": 20,
                "ip_address": "192.168.1.10",
            }
        )

    # Sequence 2: Suspicious user behavior (multiple short, low-fare rides with different drivers)
    for i in range(2):
        simulated_events.append(
            {
                "event_id": f"s2_e{i}",
                "event_timestamp": (
                    current_time - timedelta(minutes=2 * i)
                ).isoformat(),
                "event_type": "ride_completed",
                "user_id": test_user,
                "driver_id": f"driver_short_{i}",
                "fare_amount": 25000,
                "distance_km": 1.0,
                "duration_min": 3,
                "promo_code_used": "SHORT_RIDE_PROMO",
                "ip_address": f"192.168.1.1{i}",
            }
        )

    # Sequence 3: Suspicious driver behavior (many rides with few unique users)
    for i in range(6):
        simulated_events.append(
            {
                "event_id": f"s3_e{i}",
                "event_timestamp": (current_time - timedelta(minutes=i)).isoformat(),
                "event_type": "ride_completed",
                "user_id": f"user_driver_collusion_{i%2}",
                "driver_id": test_driver_B,
                "fare_amount": 30000,
                "distance_km": 1.5,
                "duration_min": 4,
                "promo_code_used": None,
                "ip_address": f"192.168.1.20",
            }
        )

    # Sequence 4: High shared rides between a specific user and driver
    for i in range(4):
        simulated_events.append(
            {
                "event_id": f"s4_e{i}",
                "event_timestamp": (current_time - timedelta(minutes=i)).isoformat(),
                "event_type": "ride_completed",
                "user_id": "user_collude_X",
                "driver_id": "driver_collude_Y",
                "fare_amount": 20000,
                "distance_km": 1.0,
                "duration_min": 3,
                "promo_code_used": "COLLUSION_PROMO",
                "ip_address": "192.168.1.30",
            }
        )

    # Process events and analyze
    print("--- Processing and Analyzing Simulated Events ---")
    for i, event in enumerate(reversed(simulated_events)):  # Process chronologically
        print(f"\nProcessing event {event['event_id']}:")
        analysis = analyzer.analyze_event(event)
        if analysis:
            # Sort analysis results for consistent output
            sorted_analysis = dict(sorted(analysis.items()))
            print(json.dumps(sorted_analysis, indent=2))
        else:
            print("No significant correlations detected for this event.")

    # Clean up
    for entity_id in [
        test_user,
        test_driver_A,
        test_driver_B,
        "user_collude_X",
        "driver_collude_Y",
        "user_driver_collusion_0",
        "user_driver_collusion_1",
        "driver_short_0",
        "driver_short_1",
    ]:
        analyzer.entity_event_stream[entity_id].clear()

    print(
        f"\nCleaned up in-memory event streams. Redis DB {redis_config_for_demo['redis_db']} not used directly by this module's history."
    )
