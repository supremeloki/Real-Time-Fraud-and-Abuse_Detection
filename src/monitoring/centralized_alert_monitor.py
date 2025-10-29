# src/monitoring/centralized_alert_monitor.py

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from src.utils.common_helpers import setup_logging

logger = setup_logging(__name__)


class CentralizedAlertMonitor:
    def __init__(self):
        self.alert_log: List[Dict[str, Any]] = []
        self.priority_map = {"critical": 5, "high": 4, "medium": 3, "low": 2, "info": 1}
        logger.info("CentralizedAlertMonitor initialized.")

    def _get_priority_value(self, priority: str) -> int:
        return self.priority_map.get(priority.lower(), self.priority_map["info"])

    def ingest_alert(
        self,
        source: str,
        message: str,
        severity: str = "info",
        details: Optional[Dict[str, Any]] = None,
    ):
        alert_entry = {
            "timestamp": datetime.now().isoformat(),
            "source": source,
            "message": message,
            "severity": severity,
            "priority_value": self._get_priority_value(severity),
            "details": details if details is not None else {},
        }
        self.alert_log.append(alert_entry)
        self.alert_log.sort(
            key=lambda x: x["priority_value"], reverse=True
        )  # Keep sorted by priority

        # Log to system log based on severity
        log_level = getattr(logging, severity.upper(), logging.INFO)
        logger.log(
            log_level,
            f"Alert from {source}: {message} (Severity: {severity}, Details: {details})",
        )

    def get_pending_alerts(self, min_severity: str = "info") -> List[Dict[str, Any]]:
        min_prio_value = self._get_priority_value(min_severity)
        return [
            alert
            for alert in self.alert_log
            if alert["priority_value"] >= min_prio_value
        ]

    def clear_alert(
        self,
        alert_timestamp: str,
        alert_source: str,
        alert_message_substring: Optional[str] = None,
    ):
        initial_count = len(self.alert_log)

        # Identify alerts to clear
        alerts_to_keep = []
        cleared_count = 0
        for alert in self.alert_log:
            match = (
                alert["timestamp"] == alert_timestamp
                and alert["source"] == alert_source
            )
            if alert_message_substring:
                match = match and (alert_message_substring in alert["message"])

            if match:
                cleared_count += 1
                logger.info(
                    f"Alert from {alert_source} at {alert_timestamp} (Msg: '{alert['message']}') cleared."
                )
            else:
                alerts_to_keep.append(alert)

        self.alert_log = alerts_to_keep
        self.alert_log.sort(
            key=lambda x: x["priority_value"], reverse=True
        )  # Re-sort after clearing

        if cleared_count == 0:
            logger.warning(
                f"No alert found matching criteria (timestamp={alert_timestamp}, source={alert_source}, message_substring={alert_message_substring}) for clearing."
            )


if __name__ == "__main__":
    import json

    monitor = CentralizedAlertMonitor()

    print("--- Ingesting Various Alerts ---")
    monitor.ingest_alert(
        "GraphAnomalyDetector",
        "High degree centrality anomaly detected for user X.",
        "critical",
        {"user_id": "userX"},
    )
    monitor.ingest_alert(
        "FeatureImpactMonitor",
        "Significant shift in 'transaction_amount' feature importance.",
        "high",
        {"feature": "transaction_amount", "change": 0.25},
    )
    monitor.ingest_alert(
        "ModelPredictionService",
        "Model inference latency spike.",
        "medium",
        {"latency_ms": 300},
    )
    monitor.ingest_alert("SystemHealth", "Database connection successful.", "info")
    monitor.ingest_alert(
        "AdaptiveThresholdManager",
        "Threshold for manual review adjusted down due to low precision.",
        "high",
        {
            "threshold_type": "manual_review_queue_score",
            "old_value": 0.5,
            "new_value": 0.49,
        },
    )
    monitor.ingest_alert(
        "RemediationOrchestrator",
        "User userY auto-blocked due to severe fraud.",
        "critical",
        {"user_id": "userY"},
    )

    print("\n--- All Pending Alerts (Sorted by Priority) ---")
    for alert in monitor.get_pending_alerts():
        print(
            f"[{alert['severity'].upper()}] {alert['source']}: {alert['message']} (Details: {alert['details']})"
        )

    print("\n--- High Severity Alerts Only ---")
    for alert in monitor.get_pending_alerts(min_severity="high"):
        print(f"[{alert['severity'].upper()}] {alert['source']}: {alert['message']}")

    print("\n--- Clearing an Alert ---")
    # Clear the first critical alert logged (GraphAnomalyDetector)
    alert_to_clear_time = monitor.alert_log[0]["timestamp"]
    alert_to_clear_source = monitor.alert_log[0]["source"]
    monitor.clear_alert(alert_to_clear_time, alert_to_clear_source)

    print("\n--- Alerts After Clearing ---")
    for alert in monitor.get_pending_alerts():
        print(f"[{alert['severity'].upper()}] {alert['source']}: {alert['message']}")

    # Ingest another alert that's similar but with different details
    monitor.ingest_alert(
        "GraphAnomalyDetector",
        "Another degree centrality anomaly detected for user Z.",
        "critical",
        {"user_id": "userZ"},
    )
    print("\n--- Alerts After Ingesting a New Similar Alert ---")
    for alert in monitor.get_pending_alerts():
        print(f"[{alert['severity'].upper()}] {alert['source']}: {alert['message']}")

    # Clear using substring
    print("\n--- Clearing an Alert using message substring ---")
    alert_to_clear_time_partial = monitor.alert_log[0][
        "timestamp"
    ]  # This will be the "userY auto-blocked" alert
    alert_to_clear_source_partial = monitor.alert_log[0]["source"]
    monitor.clear_alert(
        alert_to_clear_time_partial,
        alert_to_clear_source_partial,
        alert_message_substring="auto-blocked",
    )

    print("\n--- Alerts After Clearing with substring ---")
    for alert in monitor.get_pending_alerts():
        print(f"[{alert['severity'].upper()}] {alert['source']}: {alert['message']}")
