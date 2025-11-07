# src/decision_engine/transaction_holdout_manager.py

import logging
import json
import random
from datetime import datetime
from typing import Dict, Any, List, Set, Tuple, Callable, Optional

logger = logging.getLogger(__name__)


class TransactionHoldoutManager:
    """
    Manages a transaction holdout system to gather unbiased ground truth
    and evaluate models on real-world data without intervention.
    """

    def __init__(self, holdout_config: Dict[str, Any]):
        self.holdout_percentage = holdout_config.get(
            "percentage", 0.01
        )  # 1% of transactions
        self.holdout_reasons: Set[str] = set(
            holdout_config.get("reasons", ["unlabeled_evaluation"])
        )
        self.excluded_event_types: Set[str] = set(
            holdout_config.get("excluded_event_types", [])
        )
        self.enabled = holdout_config.get("enabled", True)
        self.holdout_log: List[Dict[str, Any]] = (
            []
        )  # In-memory log for demo, persist in real system

        # A/B test integration: if a transaction is in A/B test, it shouldn't be in holdout for basic evaluation
        self.ab_test_active_flag_func = None  # Callable to check if A/B test is active

        logger.info(
            f"TransactionHoldoutManager initialized. Holdout percentage: {self.holdout_percentage * 100:.2f}%. Enabled: {self.enabled}"
        )

    def set_ab_test_active_flag_function(self, func: Callable[[], bool]):
        """Sets a callable to check if an A/B test is currently active."""
        self.ab_test_active_flag_func = func
        logger.info("A/B test active flag function registered.")

    def _should_exclude_event(self, event: Dict[str, Any]) -> bool:
        """Checks if the event type should be excluded from holdout."""
        event_type = event.get("event_type", "").lower()
        return event_type in self.excluded_event_types

    def _is_in_ab_test(self, event: Dict[str, Any]) -> bool:
        """Checks if the event is part of an active A/B test."""
        if self.ab_test_active_flag_func and self.ab_test_active_flag_func():
            # In a real system, you'd check if this specific event's user/session is in the A/B test
            # For simplicity, we assume if A/B test is active, a percentage of traffic is already
            # being handled by it, so holdout should ideally avoid that.
            return True  # Placeholder for actual A/B test check
        return False

    def decide_holdout(
        self, event: Dict[str, Any], current_model_decision: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """
        Decides if a transaction should be put into a holdout group.

        :param event: The raw incoming event.
        :param current_model_decision: The prediction and suggested action from the fraud model.
        :return: A tuple (is_holdout, holdout_reason)
