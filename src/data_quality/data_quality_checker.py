import logging
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime


class DataQualityChecker:
    def __init__(self, ruleSet: Dict[str, Any]):
        self.rules = ruleSet
        self.issueLog: List[Dict[str, Any]] = []

    def validateRecord(self, rec: Dict[str, Any]) -> Tuple[bool, List[str]]:
        isValid = True
        errors: List[str] = []

        for fld, fldRules in self.rules.items():
            val = rec.get(fld)

            if fldRules.get("required") and val is None:
                errors.append(f"Field '{fld}' missing.")
                isValid = False
                continue

            if val is None:
                continue

            expType = fldRules.get("type")
            if expType:
                if expType == "integer" and not isinstance(val, int):
                    try:
                        val = int(val)
                    except (ValueError, TypeError):
                        errors.append(
                            f"Field '{fld}' expected int, got {type(val).__name__}."
                        )
                        isValid = False
                elif expType == "float" and not isinstance(val, (int, float)):
                    try:
                        val = float(val)
                    except (ValueError, TypeError):
                        errors.append(
                            f"Field '{fld}' expected float, got {type(val).__name__}."
                        )
                        isValid = False
                elif expType == "string" and not isinstance(val, str):
                    errors.append(
                        f"Field '{fld}' expected str, got {type(val).__name__}."
                    )
                    isValid = False
                elif expType == "boolean" and not isinstance(val, bool):
                    errors.append(
                        f"Field '{fld}' expected bool, got {type(val).__name__}."
                    )
                    isValid = False

            if fldRules.get("minVal") is not None and isinstance(val, (int, float)):
                if val < fldRules["minVal"]:
                    errors.append(
                        f"Field '{fld}' value {val} < min {fldRules['minVal']}."
                    )
                    isValid = False

            if fldRules.get("maxVal") is not None and isinstance(val, (int, float)):
                if val > fldRules["maxVal"]:
                    errors.append(
                        f"Field '{fld}' value {val} > max {fldRules['maxVal']}."
                    )
                    isValid = False

            if (
                fldRules.get("allowedVals") is not None
                and val not in fldRules["allowedVals"]
            ):
                errors.append(
                    f"Field '{fld}' value '{val}' not in allowed {fldRules['allowedVals']}."
                )
                isValid = False

            if fldRules.get("pattern") and isinstance(val, str):
                import re

                if not re.match(fldRules["pattern"], val):
                    errors.append(f"Field '{fld}' value '{val}' no match pattern.")
                    isValid = False

        if not isValid:
            self.issueLog.append(
                {
                    "ts": datetime.now().isoformat(),
                    "recId": rec.get("recordId", "N/A"),
                    "issues": errors,
                    "recSample": {k: rec.get(k) for k in list(self.rules.keys())[:5]},
                }
            )

        return isValid, errors

    def getIssueLog(self) -> List[Dict[str, Any]]:
        return self.issueLog


if __name__ == "__main__":
    import json

    schema = {
        "recordId": {"required": True, "type": "string", "pattern": "^r[0-9]{3}$"},
        "recordTs": {"required": True, "type": "string"},
        "entityId": {"required": True, "type": "string"},
        "amount": {
            "required": True,
            "type": "float",
            "minVal": 1000,
            "maxVal": 1000000,
        },
        "distance": {"required": True, "type": "float", "minVal": 0.1, "maxVal": 500},
        "payMethod": {
            "required": True,
            "type": "string",
            "allowedVals": ["card", "cash", "wallet"],
        },
        "promoCode": {"required": False, "type": "string"},
    }

    validator = DataQualityChecker(schema)

    rec1 = {
        "recordId": "r001",
        "recordTs": datetime.now().isoformat(),
        "entityId": "e123",
        "amount": 50000.0,
        "distance": 10.5,
        "payMethod": "card",
        "promoCode": "SUMMER20",
    }
    isValid, issues = validator.validateRecord(rec1)
    print(f"Rec 1 Valid: {isValid}, Issues: {issues}")

    rec2 = {
        "recordId": "r002",
        "recordTs": datetime.now().isoformat(),
        "entityId": "e456",
        "amount": 25000.0,
        "distance": 10.0,
        "payMethod": "cash",
    }
    isValid, issues = validator.validateRecord(rec2)
    print(f"Rec 2 Valid: {isValid}, Issues: {issues}")

    rec3 = {
        "recordId": "r003",
        "recordTs": datetime.now().isoformat(),
        "entityId": "e789",
        "amount": 1500.0,
        "distance": 2.0,
        "payMethod": "card",
    }
    isValid, issues = validator.validateRecord(rec3)
    print(f"Rec 3 Valid: {isValid}, Issues: {issues}")

    rec4 = {
        "recordId": "r004",
        "recordTs": datetime.now().isoformat(),
        "entityId": "e101",
        "amount": 30000.0,
        "distance": 5.0,
        "payMethod": "wallet",
    }
    isValid, issues = validator.validateRecord(rec4)
    print(f"Rec 4 Valid: {isValid}, Issues: {issues}")

    print("\n--- Issues Log ---")
    for issue in validator.getIssueLog():
        print(json.dumps(issue, indent=2))
