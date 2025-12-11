import pandas as pd
import logging
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
from src.utils.common_helpers import (
    load_config,
    setup_logging,
    serialize_json_with_datetime,
)

logger = setup_logging(__name__)


class HumanReviewSystem:
    def __init__(self, config_path: Path, env: str):
        self.config = load_config(config_path, env)
        self.logger = setup_logging(
            "HumanReviewSystem", self.config["environment"]["log_level"]
        )
        self.review_queue_file = Path(
            "./data_vault/human_review_queue.jsonl"
        )  # JSON Lines for append-only
        self.feedback_log_file = Path("./data_vault/human_feedback_log.jsonl")
        self.review_threshold = self.config["thresholds"]["action_triggers"][
            "manual_review_queue_score"
        ]
        self.logger.info(
            f"HumanReviewSystem initialized. Review threshold: {self.review_threshold}"
        )

    def add_to_review_queue(
        self, prediction_result: Dict[str, Any], raw_event: Dict[str, Any]
    ):
        if prediction_result["fraud_score"] >= self.review_threshold:
            review_entry = {
                "timestamp": datetime.now().isoformat(),
                "event_id": prediction_result["event_id"],
                "predicted_score": prediction_result["fraud_score"],
                "model_version": prediction_result["model_version"],
                "suggested_action": prediction_result["action_recommended"],
                "explanation": prediction_result.get("explanation", {}),
                "raw_event": raw_event,
                "review_status": "pending",
            }
            with open(self.review_queue_file, "a", encoding="utf-8") as f:
                f.write(serialize_json_with_datetime(review_entry) + "\n")
            self.logger.info(
                f"Event {prediction_result['event_id']} added to human review queue."
            )
            return True
        return False

    def get_pending_reviews(self) -> List[Dict[str, Any]]:
        if not self.review_queue_file.exists():
            return []

        pending_reviews = []
        with open(self.review_queue_file, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                if entry.get("review_status") == "pending":
                    pending_reviews.append(entry)
        self.logger.debug(f"Retrieved {len(pending_reviews)} pending reviews.")
        return pending_reviews

    def submit_human_feedback(
        self, event_id: str, human_decision: bool, reviewer_id: str, comments: str = ""
    ):
        feedback_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_id": event_id,
            "human_decision": human_decision,  # True if confirmed fraud, False if not fraud
            "reviewer_id": reviewer_id,
            "comments": comments,
            "feedback_processed": False,  # Mark as unprocessed for retraining pipeline
        }
        with open(self.feedback_log_file, "a", encoding="utf-8") as f:
            f.write(serialize_json_with_datetime(feedback_entry) + "\n")
        self.logger.info(
            f"Human feedback for event {event_id} submitted: Fraud={human_decision}."
        )

        self._update_review_status(event_id, "reviewed", reviewer_id)

    def _update_review_status(self, event_id: str, new_status: str, reviewer_id: str):
        # This is a simplified in-place update for demo. In production, this would involve a database.
        # For JSONL, this means rewriting the file or marking entry as superseded.
        if not self.review_queue_file.exists():
            return

        updated_lines = []
        changed = False
        with open(self.review_queue_file, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                if (
                    entry.get("event_id") == event_id
                    and entry.get("review_status") == "pending"
                ):
                    entry["review_status"] = new_status
                    entry["reviewed_by"] = reviewer_id
                    entry["reviewed_at"] = datetime.now().isoformat()
                    changed = True
                updated_lines.append(serialize_json_with_datetime(entry))

        if changed:
            with open(self.review_queue_file, "w", encoding="utf-8") as f:
                for line in updated_lines:
                    f.write(line + "\n")
            self.logger.debug(
                f"Review status for event {event_id} updated to '{new_status}'."
            )
        else:
            self.logger.warning(
                f"Event {event_id} not found or already reviewed in queue."
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Human Review System")
    parser.add_argument(
        "--env", type=str, default="dev", help="Environment (dev or prod)"
    )
    args = parser.parse_args()

    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    config_directory = project_root / "config"

    review_system = HumanReviewSystem(config_directory, args.env)

    # Example: Add an event to review queue
    sample_prediction = {
        "event_id": "test_event_001",
        "is_fraud": True,
        "fraud_score": 0.78,
        "model_version": "LGBM:v1,GNN:v1",
        "explanation": {"feature_1": 0.1, "feature_2": -0.05},
        "action_recommended": "manual_review",
    }
    sample_event = {
        "user_id": "user_xyz",
        "driver_id": "driver_abc",
        "fare_amount": 120000,
    }

    review_system.add_to_review_queue(sample_prediction, sample_event)

    # Example: Get pending reviews
    pending = review_system.get_pending_reviews()
    print(f"Pending reviews: {pending}")

    if pending:
        # Example: Submit feedback for the first pending item
        event_to_review = pending[0]["event_id"]
        review_system.submit_human_feedback(
            event_to_review,
            human_decision=True,
            reviewer_id="analyst_001",
            comments="Confirmed promo abuse pattern.",
        )

    # Check pending reviews again
    pending_after_feedback = review_system.get_pending_reviews()
    print(f"Pending reviews after feedback: {pending_after_feedback}")
