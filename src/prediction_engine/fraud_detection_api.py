from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List
import time
from src.utils.common_helpers import load_config, setup_logging
from src.prediction_engine.inference_logic import InferenceEngine

logger = setup_logging(__name__)

app = FastAPI(
    title="Snapp Fraud Detection API",
    description="Real-time API for predicting fraudulent activities.",
    version="1.0.0",
)

# Global inference engine instance
inference_engine: InferenceEngine = None


class FraudDetectionRequest(BaseModel):
    event_id: str
    user_id: str
    driver_id: str = None
    ride_id: str = None
    event_type: str
    event_timestamp: str
    start_location_lat: float = None
    start_location_lon: float = None
    end_location_lat: float = None
    end_location_lon: float = None
    fare_amount: float = None
    distance_km: float = None
    duration_min: float = None
    payment_method: str = None
    promo_code_used: str = None
    device_info: str = None
    ip_address: str = None
    app_version: str = None
    user_agent: str = None


class FraudDetectionResponse(BaseModel):
    event_id: str
    is_fraud_predicted: bool
    fraud_score: float
    model_version: str
    explanation: Dict[str, Any] = None
    action_recommended: str
    latency_ms: float


@app.on_event("startup")
async def startup_event():
    global inference_engine
    parser = argparse.ArgumentParser(description="Snapp Fraud Detection API")
    parser.add_argument(
        "--env", type=str, default="dev", help="Environment (dev or prod)"
    )
    args, _ = parser.parse_known_args()  # Use parse_known_args to ignore FastAPI's args

    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    config_directory = project_root / "config"

    try:
        inference_engine = InferenceEngine(config_directory, args.env)
        logger.info("InferenceEngine initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize InferenceEngine: {e}", exc_info=True)
        # Depending on criticality, you might want to stop the app or raise the error
        raise RuntimeError("InferenceEngine initialization failed.")


@app.post("/predict", response_model=FraudDetectionResponse)
async def predict_fraud(request: FraudDetectionRequest):
    start_time = time.perf_counter()
    if inference_engine is None:
        logger.error("InferenceEngine not initialized.")
        raise HTTPException(status_code=500, detail="Inference service not ready.")

    try:
        event_dict = request.dict(by_alias=True)

        prediction_result = inference_engine.run_inference(event_dict)

        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            f"Prediction for event {request.event_id} took {latency_ms:.2f}ms. Score: {prediction_result['fraud_score']:.4f}"
        )

        return FraudDetectionResponse(
            event_id=request.event_id,
            is_fraud_predicted=prediction_result["is_fraud"],
            fraud_score=prediction_result["fraud_score"],
            model_version=prediction_result["model_version"],
            explanation=prediction_result.get("explanation"),
            action_recommended=prediction_result["action_recommended"],
            latency_ms=latency_ms,
        )
    except Exception as e:
        logger.error(
            f"Error during fraud prediction for event {request.event_id}: {e}",
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


@app.get("/health")
async def health_check():
    if inference_engine and inference_engine.is_ready():
        return {"status": "healthy", "message": "Inference engine is ready."}
    logger.warning("Health check failed: Inference engine not ready.")
    raise HTTPException(status_code=503, detail="Inference engine not ready.")


if __name__ == "__main__":
    import uvicorn

    # This block is for direct execution of the script for testing/dev
    # In production, it would be run by a Gunicorn/Uvicorn server directly.
    parser = argparse.ArgumentParser(description="Run Snapp Fraud Detection API")
    parser.add_argument(
        "--env", type=str, default="dev", help="Environment (dev or prod)"
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the API on")
    args = parser.parse_args()

    # Manually call startup_event for direct script execution
    # This mimics what FastAPI does internally on startup
    import asyncio

    asyncio.run(startup_event())

    uvicorn.run(app, host=args.host, port=args.port)
