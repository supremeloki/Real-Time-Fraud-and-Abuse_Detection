# src/experiment_engine/ab_test_manager.py

import logging
import random
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from src.utils.common_helpers import setup_logging

logger = setup_logging(__name__)


class ABTestManager:
    def __init__(self, config: Dict[str, Any]):
        self.experiments: Dict[str, Dict[str, Any]] = config.get("experiments", {})
        self.active_experiments: Dict[str, Dict[str, Any]] = {}
        self.decision_log: List[Dict[str, Any]] = []
        self._load_active_experiments()
        logger.info(
            f"ABTestManager initialized with {len(self.experiments)} defined experiments."
        )

    def _load_active_experiments(self):
        now = datetime.now()
        for exp_name, exp_config in self.experiments.items():
            start_time_str = exp_config.get("start_time")
            end_time_str = exp_config.get("end_time")

            if start_time_str and end_time_str:
                start_time = datetime.fromisoformat(start_time_str)
                end_time = datetime.fromisoformat(end_time_str)

                if start_time <= now <= end_time:
                    self.active_experiments[exp_name] = exp_config
                    logger.info(f"Experiment '{exp_name}' is active.")
                else:
                    logger.debug(
                        f"Experiment '{exp_name}' is not active (outside time window)."
                    )
            else:
                logger.warning(
                    f"Experiment '{exp_name}' missing start/end times. Skipping activation."
                )

    def _assign_variant(
        self, experiment_config: Dict[str, Any], entity_id: str
    ) -> Optional[str]:
        variants = experiment_config.get("variants", {})
        total_traffic = experiment_config.get("total_traffic_percentage", 1.0)

        if random.random() >= total_traffic:
            return None  # Not part of the experiment's total traffic

        variant_weights = {
            v_name: v_details.get("traffic_percentage", 0.0)
            for v_name, v_details in variants.items()
        }

        # Ensure sum of weights is 1 for variants, re-normalize if needed
        sum_weights = sum(variant_weights.values())
        if sum_weights == 0:
            logger.error(
                f"Experiment has no traffic defined for variants. {experiment_config.get('name')}"
            )
            return None

        # Use a consistent hashing for deterministic assignment, or random for simple demo
        r = random.random() * sum_weights
        cumulative_weight = 0
        for variant_name, weight in variant_weights.items():
            cumulative_weight += weight
            if r <= cumulative_weight:
                return variant_name
        return None

    def get_experiment_assignment(
        self, entity_id: str, event: Dict[str, Any]
    ) -> Dict[str, Any]:
        assignments = {}
        for exp_name, exp_config in self.active_experiments.items():
            assigned_variant = self._assign_variant(exp_config, entity_id)
            if assigned_variant:
                assignments[exp_name] = assigned_variant
                self.decision_log.append(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "entity_id": entity_id,
                        "event_id": event.get("event_id"),
                        "experiment_name": exp_name,
                        "assigned_variant": assigned_variant,
                        "event_context": {
                            k: event.get(k)
                            for k in ["user_id", "driver_id", "fare_amount"]
                        },
                    }
                )
                logger.debug(
                    f"Entity {entity_id} assigned to variant {assigned_variant} for experiment {exp_name}."
                )
        return assignments

    def is_entity_in_any_active_experiment(self, entity_id: str) -> bool:
        for exp_name, exp_config in self.active_experiments.items():
            if self._assign_variant(
                exp_config, entity_id
            ):  # This implies a re-roll, not ideal for consistency
                # A proper implementation would check a persistent assignment store
                return True
        return False

    def get_decision_log(self) -> List[Dict[str, Any]]:
        return self.decision_log


if __name__ == "__main__":
    import time

    # Dummy A/B test configuration
    ab_config_demo = {
        "experiments": {
            "model_v1_vs_v2": {
                "name": "Model_V1_vs_V2_Fraud_Detection",
                "start_time": (datetime.now() - timedelta(hours=1)).isoformat(),
                "end_time": (datetime.now() + timedelta(days=7)).isoformat(),
                "total_traffic_percentage": 0.5,  # 50% of traffic for this experiment
                "variants": {
                    "control_v1": {
                        "description": "Uses Fraud Model V1",
                        "traffic_percentage": 0.5,
                    },  # 25% of total traffic
                    "test_v2": {
                        "description": "Uses Fraud Model V2",
                        "traffic_percentage": 0.5,
                    },  # 25% of total traffic
                },
            },
            "new_policy_test": {
                "name": "New_Review_Policy_Test",
                "start_time": (datetime.now() - timedelta(minutes=30)).isoformat(),
                "end_time": (datetime.now() + timedelta(days=2)).isoformat(),
                "total_traffic_percentage": 0.2,  # 20% of traffic for this experiment
                "variants": {
                    "control_old_policy": {
                        "description": "Uses existing review policy",
                        "traffic_percentage": 0.5,
                    },  # 10% of total traffic
                    "test_new_policy": {
                        "description": "Uses new aggressive review policy",
                        "traffic_percentage": 0.5,
                    },  # 10% of total traffic
                },
            },
            "inactive_experiment": {
                "name": "Should_Be_Inactive",
                "start_time": (datetime.now() - timedelta(days=2)).isoformat(),
                "end_time": (datetime.now() - timedelta(days=1)).isoformat(),
                "total_traffic_percentage": 1.0,
                "variants": {"default": {"traffic_percentage": 1.0}},
            },
        }
    }

    manager = ABTestManager(ab_config_demo)

    print("--- Simulating Events and A/B Test Assignments ---")

    for i in range(20):
        user_id = f"user_{i}"
        event_time = datetime.now() + timedelta(seconds=i * 10)
        event = {
            "event_id": f"event_{i}",
            "event_timestamp": event_time.isoformat(),
            "user_id": user_id,
            "driver_id": f"driver_{i%3}",
            "fare_amount": 50000 + i * 1000,
        }

        assignments = manager.get_experiment_assignment(user_id, event)
        if assignments:
            print(
                f"Event {event['event_id']} (User {user_id}) -> Assignments: {assignments}"
            )
        else:
            print(
                f"Event {event['event_id']} (User {user_id}) -> No experiment assignment."
            )
        time.sleep(0.1)

    print(f"\nTotal A/B test decisions logged: {len(manager.get_decision_log())}")

    print("\n--- Example of an active experiment check ---")
    some_user_id = "user_5"
    if manager.is_entity_in_any_active_experiment(some_user_id):
        print(f"User {some_user_id} is potentially in an active A/B test.")
    else:
        print(f"User {some_user_id} is not in any active A/B test.")
