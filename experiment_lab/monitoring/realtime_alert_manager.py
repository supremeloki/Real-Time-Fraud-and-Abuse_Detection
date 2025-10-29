import logging
import json
from datetime import datetime
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class RealtimeAlertManager:
    def __init__(self, config: Dict[str, Any]):
        self.alert_thresholds = config.get("alert_thresholds", {
            "critical_score": 0.95,
            "high_score": 0.75,
            "medium_score": 0.50
        })
        self.recipients = config.get("recipients", {
            "critical": ["fraud_ops_lead@example.com"],
            "high": ["fraud_ops_team@example.com"],
            "medium": ["analyst_on_duty@example.com"]
        })
        self.alert_history: List[Dict[str, Any]] = []
        logger.info("RealtimeAlertManager initialized with thresholds: %s", self.alert_thresholds)

    def _send_notification(self, level: str, message: str, event_id: str, fraud_score: float, details: Dict[str, Any]):
        notification_level = getattr(logging, level.upper())

        alert_info = {
            "timestamp": datetime.now().isoformat(),
            "level": level.upper(),
            "event_id": event_id,
            "fraud_score": fraud_score,
            "message": message,
            "recipients": self.recipients.get(level, []),
            "details": details
        }
        self.alert_history.append(alert_info)
        logger.log(notification_level, 
                   f"ALERT [{level.upper()}]: Event {event_id} (Score: {fraud_score:.2f}) - {message}. Recipients: {', '.join(self.recipients.get(level, []))}")

    def process_prediction(self, prediction_result: Dict[str, Any], raw_event: Dict[str, Any]):
        event_id = prediction_result.get("event_id", "N/A")
        fraud_score = prediction_result.get("fraud_score", 0.0)
        is_fraud = prediction_result.get("is_fraud", False)

        alert_message = f"Fraud score {fraud_score:.2f} detected for event {event_id}. Predicted fraud: {is_fraud}."
        details = {
            "user_id": raw_event.get("user_id"),
            "driver_id": raw_event.get("driver_id"),
            "fare_amount": raw_event.get("fare_amount"),
            "distance_km": raw_event.get("distance_km")
        }

        if fraud_score >= self.alert_thresholds["critical_score"]:
            self._send_notification("critical", alert_message + " Immediate action required.", event_id, fraud_score, details)
        elif fraud_score >= self.alert_thresholds["high_score"]:
            self._send_notification("warning", alert_message + " High risk event, requires review.", event_id, fraud_score, details)
        elif fraud_score >= self.alert_thresholds["medium_score"]:
            self._send_notification("info", alert_message + " Moderate risk event, monitor closely.", event_id, fraud_score, details)
        else:
            logger.info(f"Event {event_id} processed, score {fraud_score:.2f} below alert thresholds.")

    def get_alert_history(self) -> List[Dict[str, Any]]:
        return self.alert_history