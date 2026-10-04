"""Low-cardinality in-memory counters; prompts and model outputs are never stored."""

from __future__ import annotations

from threading import Lock


class GatewayMetrics:
    def __init__(self) -> None:
        self._lock = Lock()
        self._chat_requests = 0
        self._blocked_inputs = 0
        self._redacted_outputs = 0
        self._guard_latency_total_ms = 0.0
        self._guard_measurements = 0

    def record_guard(self, elapsed_ms: float) -> None:
        with self._lock:
            self._guard_latency_total_ms += elapsed_ms
            self._guard_measurements += 1

    def record_chat(self) -> None:
        with self._lock:
            self._chat_requests += 1

    def record_blocked(self) -> None:
        with self._lock:
            self._blocked_inputs += 1

    def record_redaction(self) -> None:
        with self._lock:
            self._redacted_outputs += 1

    def snapshot(self) -> dict[str, int | float]:
        with self._lock:
            average = (
                self._guard_latency_total_ms / self._guard_measurements
                if self._guard_measurements
                else 0.0
            )
            return {
                "chat_requests": self._chat_requests,
                "blocked_inputs": self._blocked_inputs,
                "redacted_outputs": self._redacted_outputs,
                "guard_measurements": self._guard_measurements,
                "average_guard_latency_ms": round(average, 3),
            }