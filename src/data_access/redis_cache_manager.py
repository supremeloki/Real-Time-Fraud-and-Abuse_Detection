import redis
import logging
import json
from typing import Dict, Any, Optional
from datetime import timedelta

# Add project root to Python path for imports
import sys
from pathlib import Path

current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

