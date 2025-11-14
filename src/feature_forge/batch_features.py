import pandas as pd
import numpy as np
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from src.utils.common_helpers import load_config, setup_logging

logger = setup_logging(__name__)


class BatchFeatureProcessor:
    def __init__(self, config_path: Path, env: str):
        self.config = load_config(config_path, env)
        self.logger = setup_logging(
            "BatchFeatureProcessor", self.config["environment"]["log_level"]
        )
        self.data_source_path = Path(
            "./data_vault/"
        )  # Placeholder for actual data lake path

    def load_historical_data(self, days_back: int = 30) -> pd.DataFrame:
        self.logger.info(f"Loading historical data for last {days_back} days.")
        # In a real scenario, this would load from a data lake (e.g., S3, HDFS)
        # For demo, let's assume 'synthetic_fraud_events.csv' contains some historical data.
        try:
            df = pd.read_csv(self.data_source_path / "synthetic_fraud_events.csv")
            df["event_timestamp"] = pd.to_datetime(df["event_timestamp"])
            cutoff_date = datetime.now() - timedelta(days=days_back)
            df = df[df["event_timestamp"] >= cutoff_date]
            self.logger.info(f"Loaded {len(df)} historical events.")
            return df
        except FileNotFoundError:
            self.logger.error(
                "Historical data file not found. Please run data_vault/fraud_pattern_simulator/generate_abuse_scenarios.py first."
            )
            return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Error loading historical data: {e}", exc_info=True)
            return pd.DataFrame()

    def compute_user_batch_features(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Computing batch features for users.")
        user_features = (
            df.groupby("user_id")
            .agg(
                user_lifetime_rides=("ride_id", "nunique"),
                user_lifetime_avg_fare=("fare_amount", "mean"),
                user_lifetime_total_promo_used=(
                    "promo_code_used",
                    lambda x: x.count() if x.notna().any() else 0,
                ),
                user_lifetime_cancellation_rate=(
                    "event_type",
                    lambda x: (
                        (x == "ride_cancelled").sum() / x.count()
