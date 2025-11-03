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

