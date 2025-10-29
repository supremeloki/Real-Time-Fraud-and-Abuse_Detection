import boto3
import pandas as pd
import logging
from io import BytesIO, StringIO
from pathlib import Path
from typing import Optional, Dict, Any
import os
import json
from collections import defaultdict

# Add project root to Python path for imports
import sys

current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


class DataLakeClient:
    def __init__(self, config: Dict[str, Any]):
        self.s3_bucket = config.get("s3_bucket_name", "snapp-ml-datalake-prod")
        self.aws_region = config.get("aws_region", "eu-central-1")

        # Initialize local data storage for testing without AWS
        self.local_storage_path = Path(
            config.get("local_storage_path", "./local_s3_data")
        )
        self.local_storage_path.mkdir(exist_ok=True)
        self.data_store: Dict[str, Dict[str, Any]] = defaultdict(
            dict
        )  # bucket -> key -> data

        # Check if real AWS credentials are available
        has_real_credentials = bool(
            os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY")
        )

        if has_real_credentials:
            try:
                self.s3_client = boto3.client("s3", region_name=self.aws_region)
                self.s3_resource = boto3.resource("s3", region_name=self.aws_region)
                self.use_local_fallback = False
                logger.info(
                    f"DataLakeClient initialized with real AWS credentials for S3 bucket '{self.s3_bucket}' in '{self.aws_region}'."
                )
            except Exception as e:
                logger.warning(
                    f"Failed to initialize S3 client with real credentials: {e}. Falling back to local storage mode."
                )
                self.s3_client = None
                self.s3_resource = None
                self.use_local_fallback = True
        else:
            logger.info(
                f"DataLakeClient initialized in local storage mode (no AWS credentials) for S3 bucket '{self.s3_bucket}' in '{self.aws_region}'."
            )
            self.s3_client = None
            self.s3_resource = None
            self.use_local_fallback = True

    def _get_s3_key(self, prefix: str, filename: str) -> str:
        return f"{prefix.strip('/')}/{filename}"

    def download_file(self, s3_prefix: str, s3_filename: str, local_path: Path) -> bool:
        s3_key = self._get_s3_key(s3_prefix, s3_filename)
        if not self.use_local_fallback:
            if self.s3_client is None:
                logger.error("S3 client not initialized. Cannot download file.")
                return False
            try:
                self.s3_client.download_file(self.s3_bucket, s3_key, str(local_path))
                logger.info(f"Downloaded '{s3_key}' to '{local_path}'.")
                return True
            except Exception as e:
                logger.error(
                    f"Failed to download '{s3_key}' from S3: {e}", exc_info=True
                )
                return False
        else:
            # Local fallback mode
            try:
                if s3_key in self.data_store[self.s3_bucket]:
                    with open(local_path, "wb") as f:
                        f.write(self.data_store[self.s3_bucket][s3_key]["data"])
                    logger.info(
                        f"Downloaded '{s3_key}' to '{local_path}' from local storage."
                    )
                    return True
                else:
                    logger.error(f"Key '{s3_key}' not found in local storage.")
                    return False
            except Exception as e:
                logger.error(
                    f"Failed to download '{s3_key}' from local storage: {e}",
                    exc_info=True,
                )
                return False

    def upload_file(self, local_path: Path, s3_prefix: str, s3_filename: str) -> bool:
        s3_key = self._get_s3_key(s3_prefix, s3_filename)
        if not self.use_local_fallback:
            if self.s3_client is None:
                logger.error("S3 client not initialized. Cannot upload file.")
                return False
            try:
                self.s3_client.upload_file(str(local_path), self.s3_bucket, s3_key)
                logger.info(f"Uploaded '{local_path}' to S3 as '{s3_key}'.")
                return True
            except Exception as e:
                logger.error(
                    f"Failed to upload '{local_path}' to S3: {e}", exc_info=True
                )
                return False
        else:
            # Local fallback mode
            try:
                with open(local_path, "rb") as f:
                    data = f.read()
                self.data_store[self.s3_bucket][s3_key] = {
                    "data": data,
                    "metadata": {
                        "size": len(data),
                        "uploaded_at": pd.Timestamp.now().isoformat(),
                    },
                }
                logger.info(f"Uploaded '{local_path}' to local storage as '{s3_key}'.")
                return True
            except Exception as e:
                logger.error(
                    f"Failed to upload '{local_path}' to local storage: {e}",
                    exc_info=True,
                )
                return False

    def load_dataframe_from_s3(
        self, s3_prefix: str, s3_filename: str, file_format: str = "csv"
    ) -> Optional[pd.DataFrame]:
        s3_key = self._get_s3_key(s3_prefix, s3_filename)
        if not self.use_local_fallback:
            if self.s3_client is None:
                logger.error("S3 client not initialized. Cannot load DataFrame.")
                return None
            try:
                obj = self.s3_client.get_object(Bucket=self.s3_bucket, Key=s3_key)
                if file_format == "csv":
                    df = pd.read_csv(StringIO(obj["Body"].read().decode("utf-8")))
                elif file_format == "parquet":
                    df = pd.read_parquet(BytesIO(obj["Body"].read()))
                elif file_format == "json":
                    df = pd.read_json(
                        StringIO(obj["Body"].read().decode("utf-8")), lines=True
                    )  # Assuming JSONL
                else:
                    raise ValueError(f"Unsupported file format: {file_format}")
                logger.info(
                    f"Loaded DataFrame from S3 key '{s3_key}'. Rows: {len(df)}."
                )
                return df
            except self.s3_client.exceptions.NoSuchKey:
                logger.warning(f"S3 key '{s3_key}' not found.")
                return None
            except Exception as e:
                logger.error(
                    f"Error loading DataFrame from S3 key '{s3_key}': {e}",
                    exc_info=True,
                )
                return None
        else:
            # Local fallback mode
            try:
                if s3_key in self.data_store[self.s3_bucket]:
                    data = self.data_store[self.s3_bucket][s3_key]["data"]
                    if file_format == "csv":
                        df = pd.read_csv(BytesIO(data))
                    elif file_format == "parquet":
                        df = pd.read_parquet(BytesIO(data))
                    elif file_format == "json":
                        df = pd.read_json(BytesIO(data), lines=True)
                    else:
                        raise ValueError(f"Unsupported file format: {file_format}")
                    logger.info(
                        f"Loaded DataFrame from local storage key '{s3_key}'. Rows: {len(df)}."
                    )
                    return df
                else:
                    logger.warning(f"Key '{s3_key}' not found in local storage.")
                    return None
            except Exception as e:
                logger.error(
                    f"Error loading DataFrame from local storage key '{s3_key}': {e}",
                    exc_info=True,
                )
                return None

    def save_dataframe_to_s3(
        self,
        df: pd.DataFrame,
        s3_prefix: str,
        s3_filename: str,
        file_format: str = "csv",
    ) -> bool:
        s3_key = self._get_s3_key(s3_prefix, s3_filename)
        if not self.use_local_fallback:
            if self.s3_client is None:
                logger.error("S3 client not initialized. Cannot save DataFrame.")
                return False
            buffer = BytesIO()
            try:
                if file_format == "csv":
                    df.to_csv(buffer, index=False)
                elif file_format == "parquet":
                    df.to_parquet(buffer, index=False)
                elif file_format == "json":
                    df.to_json(buffer, orient="records", lines=True)
                else:
                    raise ValueError(f"Unsupported file format: {file_format}")

                buffer.seek(0)
                self.s3_client.put_object(
                    Bucket=self.s3_bucket, Key=s3_key, Body=buffer.getvalue()
                )
                logger.info(f"Saved DataFrame to S3 key '{s3_key}'.")
                return True
            except Exception as e:
                logger.error(
                    f"Error saving DataFrame to S3 key '{s3_key}': {e}", exc_info=True
                )
                return False
        else:
            # Local fallback mode
            buffer = BytesIO()
            try:
                if file_format == "csv":
                    df.to_csv(buffer, index=False)
                elif file_format == "parquet":
                    df.to_parquet(buffer, index=False)
                elif file_format == "json":
                    df.to_json(buffer, orient="records", lines=True)
                else:
                    raise ValueError(f"Unsupported file format: {file_format}")

                buffer.seek(0)
                data = buffer.getvalue()
                self.data_store[self.s3_bucket][s3_key] = {
                    "data": data,
                    "metadata": {
                        "size": len(data),
                        "saved_at": pd.Timestamp.now().isoformat(),
                    },
                }
                logger.info(f"Saved DataFrame to local storage key '{s3_key}'.")
                return True
            except Exception as e:
                logger.error(
                    f"Error saving DataFrame to local storage key '{s3_key}': {e}",
                    exc_info=True,
                )
                return False


if __name__ == "__main__":
    print("DataLakeClient - Module loaded successfully")
    print("Note: Full execution requires AWS S3 access and all dependencies installed")
    print("This module is designed to run within the main fraud detection system")

    # Dummy config
    dl_config = {"s3_bucket_name": "snapp-test-bucket", "aws_region": "eu-central-1"}
    client = DataLakeClient(dl_config)

    # Check if S3 client was initialized successfully
    if client.use_local_fallback:
        print("\n--- Local Storage Mode (No AWS Credentials) ---")
        print("Using in-memory local storage for S3 operations.")
        print("Data will persist only during the session.")
    else:
        print("\n--- AWS S3 Mode ---")
        print("Using real AWS S3 for data operations.")

    # Create a dummy DataFrame
    dummy_df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "user_id": ["user_123", "user_456", "user_789"],
            "ride_distance": [5.2, 8.7, 3.1],
            "fare_amount": [45000, 78000, 32000],
        }
    )

    # Test saving and loading
    print("\n--- Testing CSV ---")
    if client.save_dataframe_to_s3(
        dummy_df, "test_data", "dummy.csv", file_format="csv"
    ):
        loaded_df = client.load_dataframe_from_s3(
            "test_data", "dummy.csv", file_format="csv"
        )
        if loaded_df is not None:
            print("Loaded CSV DataFrame:\n", loaded_df)
            print("CSV operations working correctly!")
        else:
            print("Failed to load CSV DataFrame")
    else:
        print("Failed to save CSV DataFrame")

    print("\n--- Testing Parquet ---")
    if client.save_dataframe_to_s3(
        dummy_df, "test_data", "dummy.parquet", file_format="parquet"
    ):
        loaded_df = client.load_dataframe_from_s3(
            "test_data", "dummy.parquet", file_format="parquet"
        )
        if loaded_df is not None:
            print("Loaded Parquet DataFrame:\n", loaded_df)
            print("Parquet operations working correctly!")
        else:
            print("Failed to load Parquet DataFrame")
    else:
        print("Failed to save Parquet DataFrame")

    print("\n--- Testing JSON ---")
    if client.save_dataframe_to_s3(
        dummy_df, "test_data", "dummy.json", file_format="json"
    ):
        loaded_df = client.load_dataframe_from_s3(
            "test_data", "dummy.json", file_format="json"
        )
        if loaded_df is not None:
            print("Loaded JSON DataFrame:\n", loaded_df)
            print("JSON operations working correctly!")
        else:
            print("Failed to load JSON DataFrame")
    else:
        print("Failed to save JSON DataFrame")
