import logging
import random
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from collections import deque

import numpy as np
from src.utils.common_helpers import setup_logging

logger = setup_logging(__name__)


class OperationalMetricsCollector:
    def __init__(self, buffer_size: int = 1000, window_seconds: int = 3600):
        self.buffer_size = buffer_size
        self.window_seconds = window_seconds
        self.latency_metrics: deque = deque(
            maxlen=buffer_size
        )  # list of (timestamp, latency_ms)
        self.error_metrics: deque = deque(
            maxlen=buffer_size
        )  # list of (timestamp, error_type)
        self.throughput_metrics: deque = deque(
            maxlen=buffer_size
        )  # list of (timestamp, num_events)
        self.last_throughput_flush = datetime.now()
        self.current_events_count = 0
        logger.info(
            f"OperationalMetricsCollector initialized with buffer_size={buffer_size}, window_seconds={window_seconds}."
        )

    def record_latency(self, latency_ms: float):
        self.latency_metrics.append((datetime.now(), latency_ms))
        logger.debug(f"Recorded latency: {latency_ms:.2f}ms")

    def record_error(self, error_type: str, details: Optional[Dict[str, Any]] = None):
        self.error_metrics.append(
            (datetime.now(), {"type": error_type, "details": details or {}})
        )
        logger.warning(f"Recorded error: {error_type}")

    def increment_throughput(self, count: int = 1):
        self.current_events_count += count
        logger.debug(f"Incremented throughput by {count}.")

    def _flush_throughput(self):
        now = datetime.now()
        if (
            now - self.last_throughput_flush
        ).total_seconds() >= 1:  # Flush every second for realistic throughput
            self.throughput_metrics.append((now, self.current_events_count))
            self.current_events_count = 0
            self.last_throughput_flush = now
            logger.debug("Throughput flushed.")

    def _clean_old_metrics(self):
        cutoff_time = datetime.now() - timedelta(seconds=self.window_seconds)
        for metric_deque in [
            self.latency_metrics,
            self.error_metrics,
            self.throughput_metrics,
        ]:
            while metric_deque and metric_deque[0][0] < cutoff_time:
                metric_deque.popleft()

    def get_aggregated_metrics(self) -> Dict[str, Any]:
        self._flush_throughput()
        self._clean_old_metrics()

        metrics = {}

        # Latency
        latencies = [l[1] for l in self.latency_metrics]
        if latencies:
            metrics["latency_avg_ms"] = float(np.mean(latencies))
            metrics["latency_p90_ms"] = float(np.percentile(latencies, 90))
            metrics["latency_max_ms"] = float(np.max(latencies))
            metrics["latency_count"] = len(latencies)
        else:
            metrics["latency_avg_ms"] = 0.0
            metrics["latency_p90_ms"] = 0.0
            metrics["latency_max_ms"] = 0.0
            metrics["latency_count"] = 0

        # Errors
        error_types = [e[1]["type"] for e in self.error_metrics]
        if error_types:
            unique_errors, counts = np.unique(error_types, return_counts=True)
            metrics["error_counts"] = dict(zip(unique_errors, [int(c) for c in counts]))
            metrics["total_errors"] = len(error_types)
        else:
            metrics["error_counts"] = {}
            metrics["total_errors"] = 0

        # Throughput
        throughputs = [
            t[1] for t in self.throughput_metrics
        ]  # events per second (from flushing)
        if throughputs:
            metrics["throughput_avg_eps"] = float(
                np.mean(throughputs)
            )  # events per second
            metrics["throughput_max_eps"] = float(np.max(throughputs))
            metrics["throughput_total_events"] = int(np.sum(throughputs))
        else:
            metrics["throughput_avg_eps"] = 0.0
            metrics["throughput_max_eps"] = 0.0
            metrics["throughput_total_events"] = 0

        metrics["current_window_end"] = datetime.now().isoformat()
        metrics["window_duration_seconds"] = self.window_seconds

        logger.debug("Aggregated operational metrics computed.")
        return metrics


if __name__ == "__main__":
    import json

    collector = OperationalMetricsCollector(
        buffer_size=50, window_seconds=60
    )  # 1-minute window

    print("--- Simulating Metrics Collection ---")

    # Simulate some events over a short period
    for i in range(30):
        # Latency
        collector.record_latency(random.uniform(10, 100))
        if i % 5 == 0:
            collector.record_latency(
                random.uniform(200, 500)
            )  # Simulate some higher latencies

        # Errors
        if i % 7 == 0:
            collector.record_error("ModelInferenceFailure", {"model_id": "v2"})
        if i % 13 == 0:
            collector.record_error("DatabaseConnectionError")

        # Throughput
        collector.increment_throughput()

        time.sleep(0.1)  # Simulate time passing

    print("\n--- Aggregated Metrics (1st snapshot) ---")
    metrics_snapshot_1 = collector.get_aggregated_metrics()
    print(json.dumps(metrics_snapshot_1, indent=2))

    print("\n--- Simulating more time passing for window expiration ---")
    time.sleep(65)  # Sleep for more than the 60-second window

    # Record a few more events
    for i in range(5):
        collector.record_latency(random.uniform(50, 150))
        collector.increment_throughput()
        time.sleep(0.1)

    print("\n--- Aggregated Metrics (2nd snapshot, after window expiration) ---")
    metrics_snapshot_2 = collector.get_aggregated_metrics()
    print(json.dumps(metrics_snapshot_2, indent=2))

    print(
        "\nObserve how older metrics would have been purged from the window in snapshot 2."
    )
