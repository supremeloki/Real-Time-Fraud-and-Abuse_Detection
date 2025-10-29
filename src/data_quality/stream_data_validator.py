# src/data_quality/stream_data_validator.py

import logging
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import sys
from pathlib import Path

current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.common_helpers import setup_logging

logger = logging.getLogger(__name__)


class StreamDataValidator:
    def __init__(self, validation_rules: Dict[str, Any]):
        self.rules = validation_rules
        self.data_issues_log: List[Dict[str, Any]] = []
        logger.info(
            f"StreamDataValidator initialized with {len(self.rules)} validation rules."
        )

    def validate_event(self, event: Dict[str, Any]) -> Tuple[bool, List[str]]:
        is_valid = True
        issues: List[str] = []

        for field, field_rules in self.rules.items():
            value = event.get(field)

            if field_rules.get("required") and value is None:
                issues.append(f"Field '{field}' is required but missing.")
                is_valid = False
                continue

            if (
                value is None
            ):  # If not required and missing, no further checks needed for this field
                continue

            expected_type = field_rules.get("type")
            if expected_type:
                if expected_type == "integer" and not isinstance(value, int):
                    try:
                        value = int(value)
                    except (ValueError, TypeError):
                        issues.append(
                            f"Field '{field}' expected type integer, got {type(value).__name__}."
                        )
                        is_valid = False
                elif expected_type == "float" and not isinstance(value, (int, float)):
                    try:
                        value = float(value)
                    except (ValueError, TypeError):
                        issues.append(
                            f"Field '{field}' expected type float, got {type(value).__name__}."
                        )
                        is_valid = False
                elif expected_type == "string" and not isinstance(value, str):
                    issues.append(
                        f"Field '{field}' expected type string, got {type(value).__name__}."
                    )
                    is_valid = False
                elif expected_type == "boolean" and not isinstance(value, bool):
                    issues.append(
                        f"Field '{field}' expected type boolean, got {type(value).__name__}."
                    )
                    is_valid = False

            if field_rules.get("min_value") is not None and isinstance(
                value, (int, float)
            ):
                if value < field_rules["min_value"]:
                    issues.append(
                        f"Field '{field}' value {value} is less than min {field_rules['min_value']}."
                    )
                    is_valid = False

            if field_rules.get("max_value") is not None and isinstance(
                value, (int, float)
            ):
                if value > field_rules["max_value"]:
                    issues.append(
                        f"Field '{field}' value {value} is greater than max {field_rules['max_value']}."
                    )
                    is_valid = False

            if (
                field_rules.get("allowed_values") is not None
                and value not in field_rules["allowed_values"]
            ):
                issues.append(
                    f"Field '{field}' value '{value}' not in allowed values {field_rules['allowed_values']}."
                )
                is_valid = False

            if field_rules.get("regex_pattern") and isinstance(value, str):
                import re

                if not re.match(field_rules["regex_pattern"], value):
                    issues.append(
                        f"Field '{field}' value '{value}' does not match regex pattern."
                    )
                    is_valid = False

        if not is_valid:
            self.data_issues_log.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "event_id": event.get("event_id", "N/A"),
                    "issues": issues,
                    "raw_event_sample": {
                        k: event.get(k) for k in list(self.rules.keys())[:5]
                    },  # Log first 5 fields for context
                }
            )
            logger.warning(
                f"Data validation failed for event {event.get('event_id', 'N/A')}: {'; '.join(issues)}"
            )
        else:
            logger.debug(
                f"Event {event.get('event_id', 'N/A')} passed data validation."
            )

        return is_valid, issues

    def get_data_issues_log(self) -> List[Dict[str, Any]]:
        return self.data_issues_log


if __name__ == "__main__":
    import json

    validation_schema = {
        "event_id": {
            "required": True,
            "type": "string",
            "regex_pattern": "^e[0-9]{3}$",
        },
        "event_timestamp": {"required": True, "type": "string"},
        "user_id": {"required": True, "type": "string"},
        "fare_amount": {
            "required": True,
            "type": "float",
            "min_value": 1000,
            "max_value": 1000000,
        },
        "distance_km": {
            "required": True,
            "type": "float",
            "min_value": 0.1,
            "max_value": 500,
        },
        "payment_method": {
            "required": True,
            "type": "string",
            "allowed_values": ["credit_card", "cash", "wallet"],
        },
        "promo_code_used": {"required": False, "type": "string"},
        "is_fraud_scenario": {"required": False, "type": "boolean"},
    }

    validator = StreamDataValidator(validation_schema)

    print("--- Testing Data Validation ---")

    # Valid event
    event1 = {
        "event_id": "e001",
        "event_timestamp": datetime.now().isoformat(),
        "user_id": "u123",
        "fare_amount": 50000.0,
        "distance_km": 10.5,
        "payment_method": "credit_card",
        "promo_code_used": "SUMMER20",
    }
    is_valid, issues = validator.validate_event(event1)
    print(f"Event 1 Valid: {is_valid}, Issues: {issues}")

    print("\n--- Data Issues Log ---")
    issues_log = validator.get_data_issues_log()
    if issues_log:
        print(f"Found {len(issues_log)} validation issues:")
        for issue in issues_log:
            print(f"Event {issue['event_id']}: {len(issue['issues'])} issues")
            for issue_detail in issue["issues"]:
                print(f"  - {issue_detail}")
    else:
        print("No validation issues found.")
