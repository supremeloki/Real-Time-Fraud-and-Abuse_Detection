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
                        if x.count() > 0
                        else 0
                    ),
                ),
                user_unique_driver_count=("driver_id", "nunique"),
            )
            .reset_index()
        )
        self.logger.info("User batch features computed.")
        return user_features

    def compute_driver_batch_features(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Computing batch features for drivers.")
        driver_features = (
            df.groupby("driver_id")
            .agg(
                driver_lifetime_rides=("ride_id", "nunique"),
                driver_lifetime_avg_fare=("fare_amount", "mean"),
                driver_lifetime_acceptance_rate=(
                    "event_type",
                    lambda x: (
                        (x == "ride_accepted").sum() / (x == "ride_requested").sum()
                        if (x == "ride_requested").sum() > 0
                        else 0
                    ),
                ),
                driver_lifetime_avg_distance=("distance_km", "mean"),
                driver_lifetime_unique_user_count=("user_id", "nunique"),
            )
            .reset_index()
        )
        self.logger.info("Driver batch features computed.")
        return driver_features

    def store_batch_features(
        self, user_features_df: pd.DataFrame, driver_features_df: pd.DataFrame
    ):
        self.logger.info(
            "Storing batch features (e.g., to a persistent feature store or S3)."
        )
        # In a real system, these would be pushed to a persistent feature store like DynamoDB, Cassandra, or S3
        # For demo, we'll save them as CSV files.
        user_features_df.to_csv(
            self.data_source_path / "batch_user_features.csv", index=False
        )
        driver_features_df.to_csv(
            self.data_source_path / "batch_driver_features.csv", index=False
        )
        self.logger.info("Batch features saved to CSV files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Snapp Batch Feature Processor")
    parser.add_argument(
        "--env", type=str, default="dev", help="Environment (dev or prod)"
    )
    parser.add_argument(
        "--days_back", type=int, default=30, help="Number of days for historical data."
    )
    args = parser.parse_args()

    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    config_directory = project_root / "config"

    processor = BatchFeatureProcessor(config_directory, args.env)
    historical_df = processor.load_historical_data(args.days_back)

    if not historical_df.empty:
        user_batch_feats = processor.compute_user_batch_features(historical_df)
        driver_batch_feats = processor.compute_driver_batch_features(historical_df)

        processor.store_batch_features(user_batch_feats, driver_batch_feats)
    else:
        logger.warning("No historical data to process for batch features.")
