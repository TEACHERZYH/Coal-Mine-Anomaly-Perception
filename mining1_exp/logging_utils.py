from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import sys
from typing import IO, Optional


class JsonLineFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for field in ("step_id", "run_id"):
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=True, sort_keys=True)


def configure_logging(
    level: int = logging.INFO, stream: Optional[IO[str]] = None
) -> logging.Logger:
    logger = logging.getLogger("mining1_exp")
    logger.handlers.clear()
    logger.setLevel(level)
    logger.propagate = False
    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    handler.setFormatter(JsonLineFormatter())
    logger.addHandler(handler)
    return logger
