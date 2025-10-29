import pandas as pd
import numpy as np
import lightgbm as lgb
import logging
import argparse
import mlflow
import mlflow.lightgbm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    accuracy_score,
    confusion_matrix,
)
from pathlib import Path
from src.utils.common_helpers import load_config, setup_logging

logger = setup_logging(__name__)


class LightGBMTrainer:
    def __init__(self, config_path: Path, env: str):
        self.config = load_config(config_path, env)
        self.logger = setup_logging(
            "LightGBMTrainer", self.config["environment"]["log_level"]
        )
        self.model_params = self.config["model_config"]["lightgbm_classifier"]
        self.data_path = Path("./data_vault/")
        self.mlflow_tracking_uri = self.config["environment"]["mlflow_tracking_uri"]
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        mlflow.set_experiment(f"snapp_fraud_lightgbm_{env}")
        self.logger.info(f"MLflow tracking URI: {self.mlflow_tracking_uri}")

    def load_and_prepare_data(self) -> (pd.DataFrame, pd.Series):
        self.logger.info("Loading and preparing data for LightGBM training.")
        try:
            events_df = pd.read_csv(self.data_path / "synthetic_fraud_events.csv")
            nodes_df = pd.read_csv(
                self.data_path / "graph_topology_data" / "graph_nodes.csv"
            )
            edges_df = pd.read_csv(
                self.data_path / "graph_topology_data" / "graph_edges.csv"
            )

            # Simple fraud labeling: any event linked to a 'collusion suspect' node or an 'is_fraud_edge'
            fraud_users = nodes_df[
                nodes_df["is_collusion_suspect"] & (nodes_df["node_type"] == "user")
            ]["node_id"].tolist()
            fraud_drivers = nodes_df[
                nodes_df["is_collusion_suspect"] & (nodes_df["node_type"] == "driver")
            ]["node_id"].tolist()
            fraud_edge_rides = edges_df[edges_df["is_fraud_edge"]]["ride_id"].tolist()

            events_df["is_fraud"] = (
                events_df["user_id"].isin(fraud_users)
                | events_df["driver_id"].isin(fraud_drivers)
                | events_df["ride_id"].isin(fraud_edge_rides)
            ).astype(int)

            # Feature Engineering (basic for demo)
            events_df["event_timestamp"] = pd.to_datetime(events_df["event_timestamp"])
            events_df["hour_of_day"] = events_df["event_timestamp"].dt.hour
            events_df["day_of_week"] = events_df["event_timestamp"].dt.dayofweek
            events_df["distance_per_duration"] = events_df["distance_km"] / (
                events_df["duration_min"].replace(0, 1e-6)
            )
            events_df["fare_per_km"] = events_df["fare_amount"] / (
                events_df["distance_km"].replace(0, 1e-6)
            )

            # Merge with batch features if available
            try:
                user_batch_features = pd.read_csv(
                    self.data_path / "batch_user_features.csv"
                )
                driver_batch_features = pd.read_csv(
                    self.data_path / "batch_driver_features.csv"
                )
                events_df = events_df.merge(
                    user_batch_features, on="user_id", how="left"
                )
                events_df = events_df.merge(
                    driver_batch_features, on="driver_id", how="left"
                )
            except FileNotFoundError:
                self.logger.warning(
                    "Batch feature files not found. Skipping merge. Run batch_features.py first."
                )

            # Fill NaNs created by merges for demonstration
            events_df = events_df.fillna(0)

            # Select features and target
            numerical_cols = [
                "fare_amount",
                "distance_km",
                "duration_min",
                "hour_of_day",
                "day_of_week",
                "distance_per_duration",
                "fare_per_km",
            ]
            batch_feature_cols = [col for col in events_df.columns if "lifetime" in col]

            features = numerical_cols + batch_feature_cols
            target = "is_fraud"

            X = events_df[features]
            y = events_df[target]
            self.logger.info(
                f"Data prepared with {len(features)} features and {len(X)} samples. Fraud ratio: {y.mean():.4f}"
            )
            return X, y
        except Exception as e:
            self.logger.error(f"Error loading or preparing data: {e}", exc_info=True)
            return pd.DataFrame(), pd.Series()

    def train_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ):
        self.logger.info("Starting LightGBM model training.")
        with mlflow.start_run():
            mlflow.log_params(self.model_params)

            model = lgb.LGBMClassifier(**self.model_params)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="auc",
                callbacks=[lgb.early_stopping(10, verbose=False)],
            )

            predictions = model.predict(X_val)
            probabilities = model.predict_proba(X_val)[:, 1]

            auc_score = roc_auc_score(y_val, probabilities)
            accuracy = accuracy_score(y_val, predictions)
            tn, fp, fn, tp = confusion_matrix(y_val, predictions).ravel()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0
            )

            pr_precision, pr_recall, _ = precision_recall_curve(y_val, probabilities)
            pr_auc = auc(pr_recall, pr_precision)

            mlflow.log_metrics(
                {
                    "auc_score": auc_score,
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1_score,
                    "pr_auc": pr_auc,
                }
            )
            self.logger.info(
                f"Model trained. AUC: {auc_score:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}"
            )

            mlflow.lightgbm.log_model(
                model,
                "lightgbm_fraud_model",
                registered_model_name="LightGBMFraudDetector",
            )
            self.logger.info("LightGBM model logged to MLflow.")
            return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LightGBM Fraud Detection Model")
    parser.add_argument(
        "--env", type=str, default="dev", help="Environment (dev or prod)"
    )
    args = parser.parse_args()

    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    config_directory = project_root / "config"

    trainer = LightGBMTrainer(config_directory, args.env)
    X, y = trainer.load_and_prepare_data()

    if not X.empty:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        trainer.train_model(X_train, y_train, X_val, y_val)
    else:
        logger.error("Training data is empty. Cannot train LightGBM model.")
