import yaml  # type:ignore
import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from jsonschema import validate, ValidationError  # type:ignore

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Manages loading, validation, and access to application configurations.
    Supports multiple configuration files and environment-specific overrides.
    """

    _instance = None  # Singleton instance

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, config_base_path: Path, schema_path: Optional[Path] = None):
        if not hasattr(self, "_initialized"):  # Ensure singleton init only once
            self.config_base_path = config_base_path
            self.schema_path = schema_path
            self.config: Dict[str, Any] = {}
            self.schema: Dict[str, Any] = {}
            self._initialized = True
            logger.info(f"ConfigManager initialized with base path: {config_base_path}")

            if self.schema_path and self.schema_path.exists():
                try:
                    with open(self.schema_path, "r") as f:
                        self.schema = yaml.safe_load(f)
                    logger.info("Configuration schema loaded successfully.")
                except Exception as e:
                    logger.error(
                        f"Failed to load configuration schema from {self.schema_path}: {e}",
                        exc_info=True,
                    )
                    self.schema = {}  # Ensure schema is empty if loading fails
            else:
                logger.warning(
                    "No configuration schema provided or file not found. Skipping schema validation."
                )

    def load_configs(self, env: str = "dev", config_files: List[str] = None):
        """
        Loads configurations from specified files and applies environment-specific overrides.
        """
        self.config = {}  # Reset config on reload

        if config_files is None:
            # Default important configs
            config_files = [
                "fraud_model_params.yaml",
                "threshold_settings.yaml",
            ]

        # Load base configurations
        for filename in config_files:
            file_path = self.config_base_path / filename
            if file_path.exists():
                try:
                    with open(file_path, "r") as f:
                        self.config.update(yaml.safe_load(f))
                    logger.debug(f"Loaded base config from {filename}.")
                except Exception as e:
                    logger.error(f"Error loading {filename}: {e}", exc_info=True)
            else:
                logger.warning(f"Configuration file not found: {filename}")

        # Apply environment-specific overrides
        env_file_path = self.config_base_path / "environments" / f"{env}.yaml"
        if env_file_path.exists():
            try:
                with open(env_file_path, "r") as f:
                    env_config = yaml.safe_load(f)
                    self._deep_merge_dicts(self.config, env_config)
                logger.info(f"Applied environment-specific overrides for '{env}'.")
            except Exception as e:
                logger.error(
                    f"Error loading environment config for '{env}' from {env_file_path}: {e}",
                    exc_info=True,
                )
        else:
            logger.warning(
                f"Environment config file not found: {env_file_path}. Proceeding with default/base configs."
            )

        # Validate the final configuration if a schema is available
        if self.schema:
            self.validate_config()

        logger.info(f"Final configuration loaded for environment '{env}'.")
        return self.config

    def _deep_merge_dicts(self, base_dict: Dict, merge_dict: Dict):
        """Recursively merges merge_dict into base_dict."""
        for k, v in merge_dict.items():
            if (
                k in base_dict
                and isinstance(base_dict[k], dict)
                and isinstance(v, dict)
            ):
                self._deep_merge_dicts(base_dict[k], v)
            else:
                base_dict[k] = v

    def validate_config(self):
        """
        Validates the loaded configuration against the schema.
        Raises ValidationError if validation fails.
        """
        if not self.schema:
            logger.warning("No schema loaded for configuration validation.")
            return

        try:
            validate(instance=self.config, schema=self.schema)
            logger.info("Configuration validated successfully against schema.")
        except ValidationError as e:
            logger.critical(
                f"Configuration validation failed: {e.message} at path {'/'.join(e.path)}"
            )
            raise

    def get(self, key: str, default: Any = None) -> Any:
        """Access a configuration value using a dot-separated key."""
        keys = key.split(".")
        current = self.config
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default
        return current


if __name__ == "__main__":
    from src.utils.common_helpers import setup_logging

    setup_logging("ConfigManagerDemo", level="INFO")

    # Create dummy config files for demonstration
    temp_config_path = Path("temp_conf_dir")
    temp_config_path.mkdir(exist_ok=True)
    (temp_config_path / "environments").mkdir(exist_ok=True)

    with open(temp_config_path / "fraud_model_params.yaml", "w") as f:
        f.write(
            "model_config:\n  lightgbm_classifier:\n    learning_rate: 0.1\n    num_boost_round: 100\n  gnn_collusion_detector:\n    hidden_channels: 64"
        )
    with open(temp_config_path / "threshold_settings.yaml", "w") as f:
        f.write(
            "thresholds:\n  detection_thresholds:\n    lightgbm_fraud_score:\n      high: 0.75\n  action_triggers:\n    auto_block_score: 0.9"
        )
    with open(temp_config_path / "environments" / "dev.yaml", "w") as f:
        f.write(
            "environment:\n  kafka_brokers: 'localhost:9092'\n  redis_host: 'localhost'\n  mlflow_tracking_uri: 'file://./mlruns'\nmodel_config:\n  lightgbm_classifier:\n    learning_rate: 0.05 # Override\nthresholds:\n  detection_thresholds:\n    lightgbm_fraud_score:\n      medium: 0.5"
        )  # Add new key
    with open(temp_config_path / "environments" / "prod.yaml", "w") as f:
        f.write(
            "environment:\n  kafka_brokers: 'kafka.prod.svc.cluster.local:9092'\n  redis_host: 'redis.prod.svc.cluster.local'\nthresholds:\n  action_triggers:\n    auto_block_score: 0.95 # Higher for prod\n"
        )

    # Create a dummy schema file
    schema_file = temp_config_path / "config_schema.yaml"
    with open(schema_file, "w") as f:
        f.write(
            """
type: object
properties:
  model_config:
    type: object
    properties:
      lightgbm_classifier:
        type: object
        properties:
          learning_rate: {type: number, minimum: 0.001, maximum: 1.0}
          num_boost_round: {type: integer, minimum: 10}
        required: [learning_rate]
    required: [lightgbm_classifier]
  thresholds:
    type: object
    properties:
      detection_thresholds:
        type: object
        properties:
          lightgbm_fraud_score:
            type: object
            properties:
              high: {type: number, minimum: 0.0, maximum: 1.0}
            required: [high]
        required: [lightgbm_fraud_score]
    required: [detection_thresholds]
required: [model_config, thresholds]
"""
        )

    print("--- Initializing ConfigManager and Loading Dev Config ---")
    config_manager = ConfigManager(temp_config_path, schema_file)
    dev_config = config_manager.load_configs(env="dev")
    print(f"Dev Config: {json.dumps(dev_config, indent=2)}")
    print(
        f"LGBM Learning Rate (dev): {config_manager.get('model_config.lightgbm_classifier.learning_rate')}"
    )
    print(
        f"Fraud High Threshold (dev): {config_manager.get('thresholds.detection_thresholds.lightgbm_fraud_score.high')}"
    )
    print(
        f"Fraud Medium Threshold (dev - new key): {config_manager.get('thresholds.detection_thresholds.lightgbm_fraud_score.medium')}"
    )

    print("\n--- Loading Prod Config ---")
    prod_config_manager = ConfigManager(
        temp_config_path, schema_file
    )  # Will use the same instance
    prod_config = prod_config_manager.load_configs(env="prod")
    print(f"Prod Config: {json.dumps(prod_config, indent=2)}")
    print(
        f"Kafka Brokers (prod): {prod_config_manager.get('environment.kafka_brokers')}"
    )
    print(
        f"Auto Block Score (prod): {prod_config_manager.get('thresholds.action_triggers.auto_block_score')}"
    )

    # Test validation failure (optional) - temporarily break a config file
    with open(temp_config_path / "fraud_model_params.yaml", "w") as f:
        f.write(
            "model_config:\n  lightgbm_classifier:\n    learning_rate: 10.0 # Invalid value\n"
        )
    try:
        print("\n--- Testing Config Validation Failure ---")
        config_manager.load_configs(env="dev")
    except ValidationError as e:
        print(f"Caught expected validation error: {e.message}")

    # Clean up dummy files
    import shutil

    shutil.rmtree(temp_config_path)
    print("\nCleaned up temporary config files.")
