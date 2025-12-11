# src/risk_scoring/dynamic_risk_policy_engine.py

import logging
import json
from typing import Dict, Any, List
from datetime import datetime
from src.utils.common_helpers import setup_logging  # Assuming common_helpers.py exists

logger = setup_logging(__name__)


class DynamicRiskPolicyEngine:
    def __init__(self, policy_config: Dict[str, Any]):
        self.policies = policy_config.get("policies", [])
        self.default_action = policy_config.get("default_action", "monitor")
        logger.info(
            f"DynamicRiskPolicyEngine initialized with {len(self.policies)} policies."
        )

    def _evaluate_condition(
        self, condition: Dict[str, Any], event_features: Dict[str, Any]
    ) -> bool:
        """Evaluates a single policy condition."""
        feature = condition["feature"]
        operator = condition["operator"]
        value = condition["value"]

        feature_value = event_features.get(feature)

        if feature_value is None:
            return False  # Feature not present, condition cannot be met

        if operator == ">":
            return feature_value > value
        elif operator == "<":
            return feature_value < value
        elif operator == ">=":
            return feature_value >= value
        elif operator == "<=":
            return feature_value <= value
        elif operator == "==":
            return feature_value == value
        elif operator == "!=":
            return feature_value != value
        elif operator == "in":
            return feature_value in value if isinstance(value, list) else False
        elif operator == "not in":
            return feature_value not in value if isinstance(value, list) else False
        else:
            logger.warning(f"Unsupported operator '{operator}' in policy condition.")
            return False

    def evaluate_policies(
        self, event_features: Dict[str, Any], model_prediction: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluates dynamic policies based on event features and model prediction.
        Returns the determined action and a list of triggered policies.
        """
        final_action = self.default_action
        triggered_policies: List[str] = []

        # Combine event features and model prediction for evaluation context
        evaluation_context = {**event_features, **model_prediction}

        for policy in self.policies:
            policy_name = policy.get("name", "Unnamed Policy")
            conditions = policy.get("conditions", [])
            logic = policy.get("logic", "AND").upper()  # "AND" or "OR"
            policy_action = policy.get("action", self.default_action)
            priority = policy.get(
                "priority", 0
            )  # Higher priority means it can override lower priority policies

            condition_results = [
                self._evaluate_condition(cond, evaluation_context)
                for cond in conditions
            ]

            policy_triggered = False
            if logic == "AND":
                policy_triggered = all(condition_results)
            elif logic == "OR":
                policy_triggered = any(condition_results)
            else:
                logger.warning(
                    f"Unsupported logic '{logic}' for policy '{policy_name}'. Defaulting to AND."
                )
                policy_triggered = all(condition_results)

            if policy_triggered:
                triggered_policies.append(policy_name)
                # Simple priority mechanism: later policies (higher in list for now) override if they have higher priority
                # In a real system, a more robust priority resolution would be needed.
                if (
                    policy_action == "block" and final_action != "block"
                ):  # Block always wins
                    final_action = "block"
                elif policy_action == "review" and final_action not in [
                    "block",
                    "review",
                ]:  # Review wins over monitor
                    final_action = "review"
                elif policy_action == "monitor" and final_action == self.default_action:
                    final_action = "monitor"  # Explicitly set if default action

                logger.info(
                    f"Policy '{policy_name}' triggered. Suggested action: {policy_action}"
                )

        # Final decision combines model output and policies
        # For simplicity, if a policy triggers a stronger action, it overrides the model's suggested_action
        if final_action == "block":
            model_prediction["action_recommended"] = "auto_block"
        elif (
            final_action == "review"
            and model_prediction["action_recommended"] != "auto_block"
        ):
            model_prediction["action_recommended"] = "manual_review"

        # If model score is very high and no blocking policy, still recommend block
        if (
            model_prediction["fraud_score"]
            >= evaluation_context.get(
                "thresholds.action_triggers.auto_block_score", 0.9
            )
            and final_action != "block"
        ):
            model_prediction["action_recommended"] = "auto_block"

        return {
            "policy_action": final_action,
            "triggered_policies": triggered_policies,
            "final_action_with_model": model_prediction["action_recommended"],
        }


if __name__ == "__main__":
    # Example Policy Configuration
    demo_policy_config = {
        "policies": [
            {
                "name": "Blacklisted IP Auto-Block",
                "conditions": [
                    {"feature": "is_ip_blacklisted", "operator": "==", "value": True}
                ],
                "logic": "AND",
                "action": "block",
                "priority": 100,
            },
            {
                "name": "High Fraud Score & Low Fare",
                "conditions": [
                    {"feature": "fraud_score", "operator": ">=", "value": 0.8},
                    {"feature": "fare_amount", "operator": "<", "value": 30000},
                ],
                "logic": "AND",
                "action": "review",
                "priority": 50,
            },
            {
                "name": "Promo Abuse Pattern",
                "conditions": [
                    {
                        "feature": "is_promo_on_watchlist",
                        "operator": "==",
                        "value": True,
                    },
                    {
                        "feature": "fare_per_km",
                        "operator": "<",
                        "value": 5000,
                    },  # Very cheap ride per km
                ],
                "logic": "AND",
                "action": "review",
                "priority": 60,
            },
            {
                "name": "Repeated Short Rides by Same Driver",
                "conditions": [
                    {
                        "feature": "driver_multiple_short_low_fare_rides",
                        "operator": "==",
                        "value": True,
                    },
                    {
                        "feature": "driver_low_unique_user_ratio",
                        "operator": "==",
                        "value": True,
                    },
                ],
                "logic": "AND",
                "action": "review",
                "priority": 70,
            },
        ],
        "default_action": "monitor",
    }

    policy_engine = DynamicRiskPolicyEngine(demo_policy_config)

    # --- Test Cases ---

    # 1. Event with blacklisted IP (should be blocked)
    event_features_1 = {
        "fare_amount": 50000,
        "distance_km": 10,
        "is_ip_blacklisted": True,
        "fare_per_km": 5000,
    }
    model_pred_1 = {"fraud_score": 0.2, "action_recommended": "monitor"}
    print("\n--- Test Case 1: Blacklisted IP ---")
    result_1 = policy_engine.evaluate_policies(event_features_1, model_pred_1)
    print(json.dumps(result_1, indent=2))

    # 2. Event with high fraud score and low fare (should be reviewed)
    event_features_2 = {
        "fare_amount": 25000,
        "distance_km": 5,
        "is_ip_blacklisted": False,
        "fare_per_km": 5000,
    }
    model_pred_2 = {
        "fraud_score": 0.85,
        "action_recommended": "manual_review",
    }  # Model also suggests review
    print("\n--- Test Case 2: High Fraud Score & Low Fare ---")
    result_2 = policy_engine.evaluate_policies(event_features_2, model_pred_2)
    print(json.dumps(result_2, indent=2))

    # 3. Event with suspicious promo and low fare/km (should be reviewed)
    event_features_3 = {
        "fare_amount": 15000,
        "distance_km": 5,
        "is_promo_on_watchlist": True,
        "fare_per_km": 3000,
    }
    model_pred_3 = {"fraud_score": 0.6, "action_recommended": "monitor"}
    print("\n--- Test Case 3: Promo Abuse Pattern ---")
    result_3 = policy_engine.evaluate_policies(event_features_3, model_pred_3)
    print(json.dumps(result_3, indent=2))

    # 4. Normal event, below all thresholds (should be monitored - default action)
    event_features_4 = {
        "fare_amount": 80000,
        "distance_km": 15,
        "is_ip_blacklisted": False,
        "fare_per_km": 5333,
    }
    model_pred_4 = {"fraud_score": 0.15, "action_recommended": "monitor"}
    print("\n--- Test Case 4: Normal Event ---")
    result_4 = policy_engine.evaluate_policies(event_features_4, model_pred_4)
    print(json.dumps(result_4, indent=2))

    # 5. Event with driver collusion indicators (should be reviewed)
    event_features_5 = {
        "fare_amount": 35000,
        "distance_km": 2.0,
        "fare_per_km": 17500,
        "driver_multiple_short_low_fare_rides": True,
        "driver_low_unique_user_ratio": True,
        "is_ip_blacklisted": False,
    }
    model_pred_5 = {
        "fraud_score": 0.7,
        "action_recommended": "manual_review",
    }  # Model might push it
    print("\n--- Test Case 5: Driver Collusion Indicators ---")
    result_5 = policy_engine.evaluate_policies(event_features_5, model_pred_5)
    print(json.dumps(result_5, indent=2))
