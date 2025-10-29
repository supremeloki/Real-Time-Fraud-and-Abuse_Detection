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
        """
        if not self.enabled:
            return False, None

        if self._should_exclude_event(event):
            logger.debug(
                f"Event {event.get('event_id')} excluded from holdout due to type '{event.get('event_type')}'."
            )
            return False, None

        if self._is_in_ab_test(event):
            logger.debug(
                f"Event {event.get('event_id')} is part of an A/B test. Excluding from holdout."
            )
            return False, None

        # Randomly select a percentage of events for holdout
        if random.random() < self.holdout_percentage:
            holdout_reason = random.choice(
                list(self.holdout_reasons)
            )  # Select a random reason
            self._log_holdout_event(event, current_model_decision, holdout_reason)
            logger.info(
                f"Event {event.get('event_id')} placed in holdout for reason: {holdout_reason}."
            )
            return True, holdout_reason

        return False, None

    def _log_holdout_event(
        self, event: Dict[str, Any], model_decision: Dict[str, Any], reason: str
    ):
        """Logs the event that was put into holdout."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_id": event.get("event_id"),
            "user_id": event.get("user_id"),
            "driver_id": event.get("driver_id"),
            "model_fraud_score": model_decision.get("fraud_score"),
            "model_predicted_fraud": model_decision.get("is_fraud"),
            "model_action_recommended": model_decision.get("action_recommended"),
            "holdout_reason": reason,
            "status": "pending_ground_truth",  # Will be updated by feedback loop
        }
        self.holdout_log.append(log_entry)
        logger.debug(f"Holdout event logged: {json.dumps(log_entry)}")

    def get_holdout_logs(self) -> List[Dict[str, Any]]:
        return self.holdout_log

    def retrieve_and_process_ground_truth(self):
        """
        Simulates retrieving ground truth for holdout events and updating logs.
        In a real system, this would involve a feedback loop with human review or external data.
        """
        processed_count = 0
        for entry in self.holdout_log:
            if entry["status"] == "pending_ground_truth":
                # Simulate ground truth (e.g., flip a coin, or assume actual fraud if model score was high)
                entry["true_label"] = random.choice(
                    [True, False]
                )  # Random ground truth for demo

                # More realistic: if model score was high, maybe ground truth is more likely to be true
                if entry["model_fraud_score"] > 0.7 and random.random() < 0.8:
                    entry["true_label"] = True
                elif entry["model_fraud_score"] < 0.3 and random.random() < 0.8:
                    entry["true_label"] = False

                entry["status"] = "ground_truth_obtained"
                entry["ground_truth_timestamp"] = datetime.now().isoformat()
                processed_count += 1
                logger.info(
                    f"Ground truth obtained for holdout event {entry['event_id']}: True Label = {entry['true_label']}"
                )
        return processed_count


if __name__ == "__main__":
    import json

    # Dummy holdout configuration
    holdout_conf_demo = {
        "percentage": 0.1,  # 10% holdout
        "reasons": ["unlabeled_evaluation", "new_rule_testing"],
        "excluded_event_types": ["test_event", "admin_activity"],
        "enabled": True,
    }

    holdout_manager = TransactionHoldoutManager(holdout_conf_demo)

    # Mock A/B test active function
    is_ab_test_active = False

    def mock_ab_test_flag():
        return is_ab_test_active

    holdout_manager.set_ab_test_active_flag_function(mock_ab_test_flag)

    print("--- Simulating Events for Holdout Decision ---")

    simulated_events = []
    for i in range(20):
        event = {
            "event_id": f"tx_{i}",
            "event_timestamp": datetime.now().isoformat(),
            "event_type": (
                "ride_completed" if i % 10 != 0 else "test_event"
            ),  # One excluded event
            "user_id": f"user_{i%5}",
            "driver_id": f"driver_{i%3}",
            "fare_amount": random.uniform(20000, 100000),
            "distance_km": random.uniform(1, 20),
        }
        # Simulate model predictions
        model_score = random.uniform(0.05, 0.95)
        model_decision = {
            "fraud_score": model_score,
            "is_fraud": model_score > 0.5,
            "action_recommended": (
                "auto_block"
                if model_score > 0.8
                else ("manual_review" if model_score > 0.5 else "monitor")
            ),
        }
        simulated_events.append((event, model_decision))

    for event, model_decision in simulated_events:
        is_holdout, reason = holdout_manager.decide_holdout(event, model_decision)
        if is_holdout:
            print(
                f"Event {event['event_id']} ({model_decision['fraud_score']:.2f}) -> HOLDOUT: {reason}"
            )
        else:
            print(
                f"Event {event['event_id']} ({model_decision['fraud_score']:.2f}) -> NOT HOLDOUT"
            )

    print(f"\nTotal holdout events logged: {len(holdout_manager.get_holdout_logs())}")

    print("\n--- Simulating Ground Truth Collection ---")
    processed = holdout_manager.retrieve_and_process_ground_truth()
    print(f"Processed {processed} holdout events with ground truth.")

    print("\n--- Final Holdout Logs ---")
    for log_entry in holdout_manager.get_holdout_logs():
        print(json.dumps(log_entry, indent=2))

    # Test with A/B test active
    print("\n--- Simulating Events with A/B Test Active ---")
    is_ab_test_active = True
    for i in range(5):
        event = {
            "event_id": f"tx_ab_{i}",
            "event_timestamp": datetime.now().isoformat(),
            "event_type": "ride_completed",
            "user_id": f"user_ab_{i}",
            "driver_id": f"driver_ab_{i}",
            "fare_amount": 50000,
            "distance_km": 10,
        }
        model_decision = {
            "fraud_score": 0.6,
            "is_fraud": False,
            "action_recommended": "manual_review",
        }
        is_holdout, reason = holdout_manager.decide_holdout(event, model_decision)
        if is_holdout:
            print(
                f"Event {event['event_id']} ({model_decision['fraud_score']:.2f}) -> HOLDOUT: {reason} (Unexpected!)"
            )
        else:
            print(
                f"Event {event['event_id']} ({model_decision['fraud_score']:.2f}) -> NOT HOLDOUT"
            )

    print(
        "\nNo new holdout events should be logged when A/B test is active (as per demo logic)."
    )
