"""Structured (JSON-lines) logging configuration."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path


class JsonLineFormatter(logging.Formatter):
    """Format records as one JSON object per line, including ``extra`` fields."""

    _RESERVED = set(vars(logging.makeLogRecord({})).keys()) | {"message", "asctime"}

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for k, v in record.__dict__.items():
            if k not in self._RESERVED and not k.startswith("_"):
                try:
                    json.dumps(v)
                    payload[k] = v
                except TypeError:
                    payload[k] = repr(v)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(log_file: Path | None = None, level: int = logging.INFO,
                      console: bool = True) -> logging.Logger:
    """Configure the ``qpi`` logger: human-readable console + JSON-lines file."""
    logger = logging.getLogger("qpi")
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False
    if console:
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        h.setLevel(logging.WARNING)
        logger.addHandler(h)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(JsonLineFormatter())
        logger.addHandler(fh)
    return logger


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"qpi.{name}")
