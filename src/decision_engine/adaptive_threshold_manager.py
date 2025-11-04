# src/decision_engine/adaptive_threshold_manager.py

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

# Add project root to Python path for imports
import sys
from pathlib import Path

current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data_access.redis_cache_manager import (
    RedisCacheManager,
)  # Assuming RedisCacheManager exists
from src.utils.common_helpers import setup_logging

logger = logging.getLogger(__name__)


class AdaptiveThresholdManager:
    """
    Dynamically adjusts fraud detection thresholds based on real-time performance metrics
    and operational feedback.
    """

    def __init__(
        self,
