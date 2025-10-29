# src/decision_engine/remediation_orchestrator.py
"""
RemediationOrchestrator - Handles automated responses to fraud detection events.
Manages user blocking, transaction review, IP throttling, and team notifications.
"""

import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

# Add project root to path for imports
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.common_helpers import setup_logging
from src.data_access.redis_cache_manager import RedisCacheManager
from src.monitoring.centralized_alert_monitor import CentralizedAlertMonitor

logger = setup_logging(__name__)


class RemediationOrchestrator:
    """
    Orchestrates automated remediation actions in response to fraud detection events.
    Supports multiple action types with cooldown mechanisms and comprehensive logging.
    """

    def __init__(self, config: Dict[str, Any], alert_monitor: CentralizedAlertMonitor):
        """
        Initialize the remediation orchestrator.

        Args:
            config: Configuration dictionary containing remediation actions and Redis settings
            alert_monitor: Alert monitoring system for notifications
        """
        self.remediation_actions: Dict[str, Dict[str, Any]] = config.get(
            "remediation_actions", {}
        )
        self.notification_targets: Dict[str, Dict[str, Any]] = config.get(
            "notification_targets", {}
        )
        self.alert_monitor = alert_monitor
        self.redis_client = RedisCacheManager(config["redis_config"])
        self.cooldown_period_seconds = config.get(
            "cooldown_period_seconds", 3600
        )  # Default 1 hour
        self.remediation_log_key = "remediation:log"
        logger.info(
            f"RemediationOrchestrator initialized with {len(self.remediation_actions)} actions."
        )

    def _execute_action(
        self,
        action_name: str,
        action_details: Dict[str, Any],
        event_context: Dict[str, Any],
    ) -> bool:
        """
        Execute a specific remediation action.

        Args:
            action_name: Name of the action to execute
            action_details: Details about the action (reason, etc.)
            event_context: Context from the triggering event

        Returns:
            bool: True if action executed successfully, False otherwise
        """
        action_config = self.remediation_actions.get(action_name)
        if not action_config:
            logger.error(f"Unknown remediation action: {action_name}")
            return False

        action_type = action_config.get("type")

        # Execute different action types
        if action_type == "block_user":
            user_id = event_context.get("user_id")
            if user_id:
                block_key = f"blocked:user:{user_id}"
                block_data = {
                    "reason": action_details.get("reason", "fraud"),
                    "timestamp": datetime.now().isoformat(),
                }
                self.redis_client.set_value(
                    block_key,
                    block_data,
                    ttl_seconds=action_config.get("block_ttl_seconds", 86400 * 7),
                )
                logger.critical(
                    f"User {user_id} blocked due to: {action_details.get('reason', 'fraud')}."
                )
                self.alert_monitor.ingest_alert(
                    "Remediation",
                    f"User {user_id} auto-blocked.",
                    "critical",
                    {
                        "action": "block_user",
                        "user_id": user_id,
                        "reason": action_details.get("reason"),
                    },
                )
                return True
            else:
                logger.warning(
                    f"Cannot block user, missing 'user_id' in event context for action {action_name}."
                )
                return False

        elif action_type == "flag_for_review":
            entity_id = (
                event_context.get("user_id")
                or event_context.get("driver_id")
                or event_context.get("event_id")
            )
            if entity_id:
                review_key = f"review:entity:{entity_id}"
                review_data = {
                    "reason": action_details.get("reason", "fraud_review"),
                    "timestamp": datetime.now().isoformat(),
                }
                self.redis_client.set_value(
                    review_key,
                    review_data,
                    ttl_seconds=action_config.get("review_ttl_seconds", 86400 * 30),
                )
                logger.warning(
                    f"Entity {entity_id} flagged for review: {action_details.get('reason', 'fraud_review')}."
                )
                self.alert_monitor.ingest_alert(
                    "Remediation",
                    f"Entity {entity_id} flagged for review.",
                    "high",
                    {
                        "action": "flag_for_review",
                        "entity_id": entity_id,
                        "reason": action_details.get("reason"),
                    },
                )
                return True
            else:
                logger.warning(
                    f"Cannot flag for review, no identifiable entity in context for action {action_name}."
                )
                return False

        elif action_type == "notify_slack":
            channel = action_config.get("channel")
            message = action_details.get(
                "message", f"Alert: {event_context.get('event_id')} needs attention."
            )
            logger.info(f"Sending notification to {channel}: {message}")
            self.alert_monitor.ingest_alert(
                "Remediation",
                f"Notification sent to {channel}.",
                "info",
                {"action": "notify_slack", "channel": channel, "message": message},
            )
            return True

        elif action_type == "throttle_ip":
            ip_address = event_context.get("ip_address")
            if ip_address:
                throttle_key = f"throttle:ip:{ip_address}"
                throttle_data = {
                    "limit": action_config.get("rate_limit", 5),
                    "period": action_config.get("rate_period_seconds", 60),
                }
                self.redis_client.set_value(
                    throttle_key,
                    throttle_data,
                    ttl_seconds=action_config.get("throttle_ttl_seconds", 3600),
                )
                logger.warning(
                    f"IP {ip_address} throttled to {throttle_data['limit']} requests per {throttle_data['period']}s."
                )
                self.alert_monitor.ingest_alert(
                    "Remediation",
                    f"IP {ip_address} throttled.",
                    "medium",
                    {"action": "throttle_ip", "ip": ip_address},
                )
                return True
            else:
                logger.warning(
                    f"Cannot throttle IP, missing 'ip_address' in event context for action {action_name}."
                )
                return False

        else:
            logger.error(
                f"Unsupported remediation action type: {action_type} for action {action_name}"
            )
            return False

    def trigger_remediation(
        self, event_id: str, suggested_action: str, event_context: Dict[str, Any]
    ):
        """
        Trigger remediation actions based on fraud detection results.

        Args:
            event_id: Unique identifier for the event
            suggested_action: The recommended remediation action
            event_context: Context information about the event
        """
        remediation_config = self.remediation_actions.get(suggested_action)
        if not remediation_config:
            logger.info(f"No remediation config found for action '{suggested_action}'.")
            return

        cooldown_key = f"cooldown:{suggested_action}:{event_id}"

        # Check cooldown
        last_triggered = self.redis_client.get_value(cooldown_key)
        if last_triggered:
            last_ts = datetime.fromisoformat(last_triggered["timestamp"])
            if (
                datetime.now() - last_ts
            ).total_seconds() < self.cooldown_period_seconds:
                logger.info(
                    f"Action '{suggested_action}' for event {event_id} on cooldown."
                )
                return

        logger.info(
            f"Initiating remediation for event {event_id} with action '{suggested_action}'."
        )

        # Execute primary action
        primary_action_name = remediation_config.get("primary_action")
        primary_action_details = {"reason": suggested_action}

        action_success = False
        if primary_action_name:
            action_success = self._execute_action(
                primary_action_name, primary_action_details, event_context
            )
            if action_success:
                # Set cooldown
                cooldown_data = {"timestamp": datetime.now().isoformat()}
                self.redis_client.set_value(
                    cooldown_key,
                    cooldown_data,
                    ttl_seconds=self.cooldown_period_seconds,
                )

                # Log successful remediation
                log_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "event_id": event_id,
                    "suggested_action": suggested_action,
                    "executed_action": primary_action_name,
                    "status": "success",
                    "context": event_context,
                }
                self.redis_client.rpush(self.remediation_log_key, log_entry)
            else:
                # Log failed remediation
                log_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "event_id": event_id,
                    "suggested_action": suggested_action,
                    "executed_action": primary_action_name,
                    "status": "failed",
                    "error": "Primary action failed",
                    "context": event_context,
                }
                self.redis_client.rpush(self.remediation_log_key, log_entry)
        else:
            logger.warning(
                f"No primary action defined for suggested action: {suggested_action}."
            )

        # Execute secondary actions
        for secondary_action in remediation_config.get("secondary_actions", []):
            self._execute_action(
                secondary_action.get("name"),
                secondary_action.get("details", {}),
                event_context,
            )

    def get_remediation_logs(self) -> List[Dict[str, Any]]:
        """
        Retrieve remediation logs from Redis.

        Returns:
            List of remediation log entries
        """
        return self.redis_client.lrange(self.remediation_log_key)


if __name__ == "__main__":
    print("RemediationOrchestrator - Module loaded successfully")
    print("Note: Full execution requires Redis running. Using in-memory mode for demo.")

    # Redis configuration (will fail gracefully if Redis not available)
    redis_conf = {
        "redis_host": "localhost",
        "redis_port": 6379,
        "redis_db": 6,
        "default_ttl_seconds": 3600,
    }

    # Mock alert monitor
    class MockAlertMonitor(CentralizedAlertMonitor):
        def __init__(self):
            super().__init__()
            self.alerts = []

        def ingest_alert(
            self,
            source: str,
            message: str,
            severity: str = "info",
            details: Optional[Dict[str, Any]] = None,
        ):
            alert_entry = {
                "source": source,
                "message": message,
                "severity": severity,
                "details": details,
            }
            self.alerts.append(alert_entry)
            print(f"[MOCK ALERT] {severity.upper()} from {source}: {message}")

    mock_alert_monitor = MockAlertMonitor()

    # Remediation configuration
    remediation_config_demo = {
        "remediation_actions": {
            "block_user_permanent": {
                "type": "block_user",
                "block_ttl_seconds": 86400 * 365,
            },
            "flag_transaction_for_review": {
                "type": "flag_for_review",
                "review_ttl_seconds": 86400 * 30,
            },
            "notify_fraud_team": {"type": "notify_slack", "channel": "#fraud-alerts"},
            "throttle_suspicious_ip": {
                "type": "throttle_ip",
                "rate_limit": 2,
                "rate_period_seconds": 60,
                "throttle_ttl_seconds": 3600,
            },
        },
        "remediation_strategies": {
            "auto_block": {
                "primary_action": "block_user_permanent",
                "secondary_actions": [
                    {
                        "name": "notify_fraud_team",
                        "details": {
                            "message": "User auto-blocked due to severe fraud."
                        },
                    }
                ],
            },
            "manual_review": {
                "primary_action": "flag_transaction_for_review",
                "secondary_actions": [
                    {
                        "name": "notify_fraud_team",
                        "details": {
                            "message": "Transaction flagged for manual review."
                        },
                    }
                ],
            },
            "potential_ip_abuse": {
                "primary_action": "throttle_suspicious_ip",
                "secondary_actions": [],
            },
        },
        "redis_config": redis_conf,
        "cooldown_period_seconds": 5,
    }

    orchestrator = RemediationOrchestrator(remediation_config_demo, mock_alert_monitor)
    orchestrator.remediation_actions.update(
        remediation_config_demo["remediation_strategies"]
    )

    # Check Redis connection (continue even if failed)
    try:
        orchestrator.redis_client.check_connection()
        print("Redis connected successfully.")
    except Exception as e:
        print(f"Redis connection failed: {e}. Continuing in demo mode.")

    print("\n--- Testing Remediation Orchestrator ---")

    # Test scenarios
    event_context_block = {
        "event_id": "tx_block_001",
        "user_id": "fraudster_A",
        "ip_address": "1.2.3.4",
    }
    event_context_review = {
        "event_id": "tx_review_001",
        "user_id": "suspicious_B",
        "ip_address": "5.6.7.8",
    }
    event_context_throttle = {
        "event_id": "tx_throttle_001",
        "user_id": "normal_C",
        "ip_address": "192.168.1.10",
    }

    # Test auto-block
    print("\n--- Triggering Auto-Block ---")
    orchestrator.trigger_remediation(
        event_context_block["event_id"], "auto_block", event_context_block
    )
    orchestrator.trigger_remediation(
        event_context_block["event_id"], "auto_block", event_context_block
    )  # Should be on cooldown

    # Test manual review
    print("\n--- Triggering Manual Review ---")
    orchestrator.trigger_remediation(
        event_context_review["event_id"], "manual_review", event_context_review
    )

    # Test IP throttling
    print("\n--- Triggering IP Throttle ---")
    orchestrator.trigger_remediation(
        event_context_throttle["event_id"], "potential_ip_abuse", event_context_throttle
    )

    # Wait for cooldown
    print("\n--- Waiting for cooldown to expire ---")
    time.sleep(remediation_config_demo["cooldown_period_seconds"] + 1)

    # Test auto-block again after cooldown
    print("\n--- Triggering Auto-Block again after cooldown ---")
    orchestrator.trigger_remediation(
        event_context_block["event_id"], "auto_block", event_context_block
    )

    # Show remediation logs
    print("\n--- Remediation Logs ---")
    logs = orchestrator.get_remediation_logs()
    for log in logs:
        print(
            f"Event {log.get('event_id')}: {log.get('executed_action')} - {log.get('status')}"
        )

    # Show Redis states (will be None if Redis not connected)
    print("\n--- Redis States ---")
    print(
        f"Blocked user fraudster_A: {orchestrator.redis_client.get_value('blocked:user:fraudster_A')}"
    )
    print(
        f"Review entity tx_review_001: {orchestrator.redis_client.get_value('review:entity:tx_review_001')}"
    )
    print(
        f"Throttled IP 192.168.1.10: {orchestrator.redis_client.get_value('throttle:ip:192.168.1.10')}"
    )

    # Cleanup
    try:
        orchestrator.redis_client.redis_client.flushdb()
        print(f"\nRedis DB {redis_conf['redis_db']} flushed for cleanup.")
    except Exception as e:
        print(f"Cleanup failed: {e}")
