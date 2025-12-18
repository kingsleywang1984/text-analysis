from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Dict


class JsonFormatter(logging.Formatter):
    """Emit log records as JSON for easier ingestion in CloudWatch."""

    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        payload: Dict[str, Any] = {
            "timestamp": datetime.now(tz=timezone.utc)
            .isoformat(timespec="milliseconds"),
            "level": record.levelname.lower(),
            "logger": record.name,
            "event": record.getMessage(),
        }

        extra = getattr(record, "log_extra", None)
        if isinstance(extra, dict):
            payload.update(extra)

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False)


def _build_logger() -> logging.Logger:
    logger = logging.getLogger("text_insight_clustering_service")
    if logger.handlers:
        return logger

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


_LOGGER = _build_logger()


def log_info(event: str, **fields: Any) -> None:
    _LOGGER.info(event, extra={"log_extra": fields})


def log_warning(event: str, **fields: Any) -> None:
    _LOGGER.warning(event, extra={"log_extra": fields})


def log_error(event: str, **fields: Any) -> None:
    _LOGGER.error(event, extra={"log_extra": fields})
