"""Validated request and response contracts for the public API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str = Field(max_length=32_000)


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = Field(default="demo-model", min_length=1, max_length=200)
    messages: list[ChatMessage] = Field(min_length=1, max_length=100)
    stream: bool = False
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_tokens: int | None = Field(default=None, ge=1, le=32_000)


class TextScanRequest(BaseModel):
    text: str = Field(min_length=1, max_length=32_000)


class PolicyViolation(BaseModel):
    error: dict[str, Any]


class ScanResponse(BaseModel):
    blocked: bool
    risk_score: float
    findings: list[str]
    semantic_similarity: float
    classifier_score: float


class HealthResponse(BaseModel):
    status: Literal["ok"]
    mode: Literal["demo", "remote"]
    semantic_enabled: bool
    classifier_ready: bool