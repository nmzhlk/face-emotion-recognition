import logging
import os
import sys
from typing import Any, Dict

from pythonjsonlogger import json


class CustomJsonFormatter(json.JsonFormatter):
    def add_fields(
        self,
        log_record: Dict[str, Any],
        record: logging.LogRecord,
        message_dict: Dict[str, Any],
    ) -> None:
        super().add_fields(log_record, record, message_dict)
        log_record["level"] = record.levelname
        log_record["logger"] = record.name
        log_record["timestamp"] = self.formatTime(
            record, "%Y-%m-%dT%H:%M:%S.%fZ"
        )
        if not log_record.get("service"):
            log_record["service"] = os.environ.get("SERVICE_NAME", "unknown")


def setup_logging(service_name: str = "None") -> None:
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)
    formatter = CustomJsonFormatter(
        "%(timestamp)s %(level)s %(logger)s %(service)s %(message)s"
    )
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    if service_name:
        logging.getLogger().info(
            "Logging configured", extra={"service": service_name}
        )
