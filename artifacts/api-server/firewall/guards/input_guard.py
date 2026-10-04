"""Composable input screening pipeline."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from firewall.config import Settings
from firewall.detectors.classifier import PromptClassifier
from firewall.detectors.rules import HeuristicDetector
from firewall.detectors.semantic import MiniLMSimilarityDetector


@dataclass(frozen=True, slots=True)
class InputDecision:
    blocked: bool
    risk_score: float
    findings: tuple[str, ...]
    semantic_similarity: float
    classifier_score: float


class InputGuard:
    """Run cheap deterministic checks first and optional models afterward."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.heuristics = HeuristicDetector()
        self.classifier = PromptClassifier(settings.classifier_model_path)
        self.semantic = MiniLMSimilarityDetector(
            model_name=settings.semantic_model,
            threshold=settings.semantic_block_threshold,
        )

    async def initialize(self) -> None:
        self.classifier.load_or_train()
        if self.settings.semantic_enabled:
            await self.semantic.initialize()

    async def scan(self, text: str) -> InputDecision:
        rule_scan = self.heuristics.scan(text)
        classifier_score = await asyncio.to_thread(
            self.classifier.attack_score,
            text,
        )
        semantic_score = (
            await self.semantic.score(text)
            if self.settings.semantic_enabled
            else 0.0
        )

        findings = list(rule_scan.findings)
        if classifier_score >= self.settings.classifier_block_threshold:
            findings.append("classifier:high_risk")
        if (
            self.settings.semantic_enabled
            and semantic_score >= self.settings.semantic_block_threshold
        ):
            findings.append("semantic:high_similarity")

        return InputDecision(
            blocked=bool(findings),
            risk_score=max(
                classifier_score,
                semantic_score,
                1.0 if rule_scan.blocked else 0.0,
            ),
            findings=tuple(sorted(set(findings))),
            semantic_similarity=semantic_score,
            classifier_score=classifier_score,
        )