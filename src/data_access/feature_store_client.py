# src/data_access/feature_store_client.py

import logging
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

# Add project root to Python path for imports
import sys
from pathlib import Path

current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data_access.redis_cache_manager import (
    RedisCacheManager,
)  # Assuming RedisCacheManager
from src.data_access.data_lake_client import DataLakeClient  # Assuming DataLakeClient
from src.feature_forge.realtime_feature_engineer import (
    RealtimeFeatureEngineer,
)  # For on-the-fly feature calculation
from src.graph_processor.node_embedding_updater import (
    NodeEmbeddingUpdater,
)  # For GNN embeddings
