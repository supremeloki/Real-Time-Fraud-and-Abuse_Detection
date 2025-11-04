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

