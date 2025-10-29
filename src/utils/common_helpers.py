import yaml
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import json
from datetime import datetime


def load_config(config_path: Path, env: str = "dev"):
    with open(config_path / "environments" / f"{env}.yaml", "r") as f:
        env_config = yaml.safe_load(f)

    with open(config_path / "fraud_model_params.yaml", "r") as f:
        model_params = yaml.safe_load(f)

    with open(config_path / "threshold_settings.yaml", "r") as f:
        thresholds = yaml.safe_load(f)

    config = {
        **env_config,
        "model_config": {**env_config.get("model_config", {}), **model_params},
        "thresholds": {**env_config.get("thresholds", {}), **thresholds},
    }
    return config


def setup_logging(name: str, log_level: str = "INFO", log_file: Path = None):
    logger = logging.getLogger(name)
    logger.setLevel(log_level.upper())

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            fh = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    return logger


class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return json.JSONEncoder.default(self, obj)


def serialize_json_with_datetime(data):
    return json.dumps(data, cls=DateTimeEncoder)


def deserialize_json_with_datetime(json_str):
    return json.loads(json_str)
