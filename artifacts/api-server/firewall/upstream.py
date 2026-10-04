"""Demo and OpenAI-compatible upstream adapters."""

from __future__ import annotations

from typing import Any, Protocol

import httpx

from firewall.config import Settings


class UpstreamError(RuntimeError):
    """An upstream failed without exposing its response body or credentials."""


class Upstream(Protocol):
    async def complete(self, payload: dict[str, Any]) -> dict[str, Any]: ...

    async def close(self) -> None: ...


class DemoUpstream:
    """Deterministic local model substitute for development and tests."""

    async def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        model = str(payload.get("model", "demo-model"))
        return {
            "id": "chatcmpl-demo",
            "object": "chat.completion",
            "created": 0,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": (
                            "Demo upstream completed the request. "
                            "Configure UPSTREAM_MODE=remote to connect a model."
                        ),
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }

    async def close(self) -> None:
        return None


class OpenAICompatibleUpstream:
    """Proxy to a fixed, operator-configured OpenAI-compatible endpoint."""

    def __init__(self, settings: Settings) -> None:
        if not settings.upstream_api_key:
            raise ValueError(
                "UPSTREAM_MODE=remote requires UPSTREAM_API_KEY to be configured."
            )
        if not settings.upstream_url.startswith("https://"):
            raise ValueError(
                "Remote upstream URLs must use HTTPS to protect prompts "
                "and credentials."
            )
        self.url = settings.upstream_url
        self.api_key = settings.upstream_api_key
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.upstream_timeout_seconds),
            follow_redirects=False,
        )

    async def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = await self.client.post(
                self.url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, ValueError) as error:
            raise UpstreamError(
                "The configured model upstream did not return a valid response."
            ) from error
        if not isinstance(data, dict):
            raise UpstreamError(
                "The configured model upstream returned an invalid response."
            )
        return data

    async def close(self) -> None:
        await self.client.aclose()


def create_upstream(settings: Settings) -> Upstream:
    if settings.upstream_mode == "demo":
        return DemoUpstream()
    return OpenAICompatibleUpstream(settings)