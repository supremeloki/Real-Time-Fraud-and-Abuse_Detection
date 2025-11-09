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
