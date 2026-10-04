"""Composition root for policy checks, metrics, and the model adapter."""

from __future__ import annotations

from typing import Any

from firewall.config import Settings
from firewall.guards.input_guard import InputDecision, InputGuard
from firewall.guards.output_guard import OutputGuard
from firewall.metrics import GatewayMetrics
from firewall.upstream import Upstream, create_upstream


class Gateway:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.input_guard = InputGuard(settings)
        self.output_guard = OutputGuard(redact_urls=settings.redact_urls)
        self.metrics = GatewayMetrics()
        self.upstream: Upstream = create_upstream(settings)

    async def initialize(self) -> None:
        await self.input_guard.initialize()

    async def close(self) -> None:
        await self.upstream.close()

    async def complete(
        self,
        payload: dict[str, Any],
    ) -> tuple[dict[str, Any], list[str]]:
        if payload.get("stream") is True:
            response = await self.upstream.complete_stream(payload)
        else:
            response = await self.upstream.complete(payload)
        redaction_findings: set[str] = set()

        def sanitize_value(value: Any) -> Any:
            if isinstance(value, str):
                result = self.output_guard.redact(value)
                redaction_findings.update(result.findings)
                return result.text
            if isinstance(value, list):
                return [sanitize_value(item) for item in value]
            if isinstance(value, dict):
                return {
                    key: sanitize_value(item)
                    for key, item in value.items()
                }
            return value

        for choice in response.get("choices", []):
            if isinstance(choice, dict):
                sanitized_choice = sanitize_value(choice)
                choice.clear()
                choice.update(sanitized_choice)

        if redaction_findings:
            self.metrics.record_redaction()
        return response, sorted(redaction_findings)


def decision_to_dict(decision: InputDecision) -> dict[str, Any]:
    return {
        "blocked": decision.blocked,
        "risk_score": round(decision.risk_score, 4),
        "findings": list(decision.findings),
        "semantic_similarity": round(decision.semantic_similarity, 4),
        "classifier_score": round(decision.classifier_score, 4),
    }