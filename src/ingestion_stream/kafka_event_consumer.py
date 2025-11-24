import json
import logging
import argparse
from pathlib import Path
import time
import os
from confluent_kafka import Consumer, KafkaException, Message  # type:ignore
from confluent_kafka.schema_registry import SchemaRegistryClient  # type:ignore
from confluent_kafka.schema_registry.avro import AvroDeserializer  # type:ignore
from src.utils.common_helpers import load_config, setup_logging  # type:ignore

logger = setup_logging(__name__)


class KafkaEventConsumer:
    def __init__(self, config_path: Path, env: str):
        self.config = load_config(config_path, env)
        self.logger = setup_logging(
            "KafkaConsumer", self.config["environment"]["log_level"]
        )

        consumer_conf = {
            "bootstrap.servers": self.config["environment"]["kafka_brokers"],
            "group.id": f"snapp-fraud-consumer-{env}",
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
        }
        self.consumer = Consumer(consumer_conf)
        self.topic = self.config["environment"]["kafka_input_topic"]

        self.schema_registry_client = SchemaRegistryClient(
            {"url": os.getenv("SCHEMA_REGISTRY_URL", "http://localhost:8081")}
        )
        self.avro_deserializer = AvroDeserializer(self.schema_registry_client)

        self.logger.info(f"Kafka consumer initialized for topic: {self.topic}")

    def start_consuming(self, message_handler):
        try:
            self.consumer.subscribe([self.topic])
            self.logger.info(
                f"Subscribed to topic '{self.topic}'. Waiting for messages..."
            )

            while True:
                msg = self.consumer.poll(timeout=1.0)
                if msg is None:
                    continue
                if msg.error():
                    if msg.error().code() == KafkaException._PARTITION_EOF:
                        continue
                    else:
                        self.logger.error(f"Kafka error: {msg.error()}")
                        break

                try:
                    # Assuming message value is Avro encoded
                    event_data = self.avro_deserializer(None, msg.value())
                    # Fallback to JSON if Avro fails or not applicable
                    if event_data is None:
                        event_data = json.loads(msg.value().decode("utf-8"))

                    self.logger.debug(
                        f"Received message: {event_data.get('event_id', 'N/A')}"
                    )
                    message_handler(event_data)
                    self.consumer.commit(message=msg)

                except json.JSONDecodeError:
