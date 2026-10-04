"""Privacy-conscious JSON audit logging to stdout."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any


class JsonAuditFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        audit_record: dict[str, Any] = getattr(record, "audit_record", {})
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **audit_record,
        }
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def get_audit_logger() -> logging.Logger:
    logger = logging.getLogger("llm_firewall.audit")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonAuditFormatter())
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger