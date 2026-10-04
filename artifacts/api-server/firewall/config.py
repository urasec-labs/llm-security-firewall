"""Environment-backed settings for the security gateway."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal


UpstreamMode = Literal["demo", "remote"]
UpstreamProvider = Literal["openai", "groq", "together", "custom"]

PROVIDER_BASE_URLS: dict[UpstreamProvider, str] = {
    "openai": "https://api.openai.com/v1",
    "groq": "https://api.groq.com/openai/v1",
    "together": "https://api.together.xyz/v1",
    "custom": "https://example.invalid/v1",
}


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime configuration; upstream destinations are operator-controlled."""

    upstream_mode: UpstreamMode = "demo"
    upstream_provider: UpstreamProvider = "openai"
    upstream_base_url: str = PROVIDER_BASE_URLS["openai"]
    upstream_api_key: str | None = None
    upstream_timeout_seconds: float = 30.0
    max_upstream_response_bytes: int = 2_097_152
    max_body_bytes: int = 1_048_576
    max_prompt_chars: int = 32_000
    classifier_model_path: str | None = None
    classifier_block_threshold: float = 0.92
    semantic_enabled: bool = False
    semantic_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    semantic_block_threshold: float = 0.86
    redact_urls: bool = True
    auth_required: bool = False
    client_api_key: str | None = None
    rate_limit_requests: int = 60
    rate_limit_window_seconds: float = 60.0
    rate_limit_max_keys: int = 10_000

    @classmethod
    def from_env(cls) -> Settings:
        mode = os.getenv("UPSTREAM_MODE", "demo").strip().lower()
        if mode not in {"demo", "remote"}:
            raise ValueError("UPSTREAM_MODE must be either 'demo' or 'remote'.")
        provider = os.getenv("UPSTREAM_PROVIDER", "openai").strip().lower()
        if provider not in PROVIDER_BASE_URLS:
            raise ValueError(
                "UPSTREAM_PROVIDER must be openai, groq, together, or custom."
            )

        timeout = float(os.getenv("UPSTREAM_TIMEOUT_SECONDS", "30"))
        max_upstream_response_bytes = int(
            os.getenv("MAX_UPSTREAM_RESPONSE_BYTES", "2097152")
        )
        max_body_bytes = int(os.getenv("MAX_BODY_BYTES", "1048576"))
        max_prompt_chars = int(os.getenv("MAX_PROMPT_CHARS", "32000"))
        classifier_threshold = float(
            os.getenv("CLASSIFIER_BLOCK_THRESHOLD", "0.92")
        )
        semantic_threshold = float(os.getenv("SEMANTIC_BLOCK_THRESHOLD", "0.86"))
        rate_limit_requests = int(os.getenv("RATE_LIMIT_REQUESTS", "60"))
        rate_limit_window_seconds = float(
            os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60")
        )
        rate_limit_max_keys = int(os.getenv("RATE_LIMIT_MAX_KEYS", "10000"))

        if timeout <= 0:
            raise ValueError("UPSTREAM_TIMEOUT_SECONDS must be greater than zero.")
        if max_upstream_response_bytes <= 0:
            raise ValueError(
                "MAX_UPSTREAM_RESPONSE_BYTES must be greater than zero."
            )
        if max_body_bytes <= 0 or max_prompt_chars <= 0:
            raise ValueError("Request size limits must be greater than zero.")
        if not 0 < classifier_threshold <= 1:
            raise ValueError("CLASSIFIER_BLOCK_THRESHOLD must be in (0, 1].")
        if not 0 < semantic_threshold <= 1:
            raise ValueError("SEMANTIC_BLOCK_THRESHOLD must be in (0, 1].")
        if (
            rate_limit_requests <= 0
            or rate_limit_window_seconds <= 0
            or rate_limit_max_keys <= 0
        ):
            raise ValueError("Rate-limit settings must be greater than zero.")

        return cls(
            upstream_mode=mode,  # type: ignore[arg-type]
            upstream_provider=provider,  # type: ignore[arg-type]
            upstream_base_url=os.getenv(
                "UPSTREAM_BASE_URL",
                PROVIDER_BASE_URLS[provider],  # type: ignore[index]
            ),
            upstream_api_key=os.getenv("UPSTREAM_API_KEY") or None,
            upstream_timeout_seconds=timeout,
            max_upstream_response_bytes=max_upstream_response_bytes,
            max_body_bytes=max_body_bytes,
            max_prompt_chars=max_prompt_chars,
            classifier_model_path=os.getenv("CLASSIFIER_MODEL_PATH") or None,
            classifier_block_threshold=classifier_threshold,
            semantic_enabled=_as_bool(os.getenv("SEMANTIC_ENABLED"), False),
            semantic_model=os.getenv(
                "SEMANTIC_MODEL",
                "sentence-transformers/all-MiniLM-L6-v2",
            ),
            semantic_block_threshold=semantic_threshold,
            redact_urls=_as_bool(os.getenv("REDACT_URLS"), True),
            auth_required=_as_bool(os.getenv("AUTH_REQUIRED"), True),
            client_api_key=os.getenv("CLIENT_API_KEY") or None,
            rate_limit_requests=rate_limit_requests,
            rate_limit_window_seconds=rate_limit_window_seconds,
            rate_limit_max_keys=rate_limit_max_keys,
        )