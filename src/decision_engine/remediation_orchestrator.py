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
