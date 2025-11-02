import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from scipy.stats import ks_2samp, chi2_contingency

# Add project root to Python path for imports
import sys
from pathlib import Path

current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.common_helpers import setup_logging
from src.data_access.data_lake_client import DataLakeClient

logger = logging.getLogger(__name__)


class DataDriftDetector:
    def __init__(self, config: Dict[str, Any]):
        self.data_lake_client = DataLakeClient(config["data_lake_config"])
        self.reference_data_path = config["data_drift_config"]["reference_data_path"]
        self.current_data_log_path = config["data_drift_config"][
