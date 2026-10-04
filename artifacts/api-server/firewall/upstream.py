"""Demo and OpenAI-compatible upstream adapters."""

from __future__ import annotations

import json
from typing import Any, Protocol
from urllib.parse import urlsplit, urlunsplit

import httpx

from firewall.config import Settings


class UpstreamError(RuntimeError):
    """An upstream failed without exposing its response body or credentials."""

    def __init__(
        self,
        message: str,
        upstream_status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.upstream_status_code = upstream_status_code


class Upstream(Protocol):
    async def complete(self, payload: dict[str, Any]) -> dict[str, Any]: ...

    async def complete_stream(self, payload: dict[str, Any]) -> dict[str, Any]: ...

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

    async def complete_stream(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self.complete(payload)

    async def close(self) -> None:
        return None


class OpenAICompatibleUpstream:
    """Proxy to a fixed, operator-configured OpenAI-compatible endpoint."""

    def __init__(
        self,
        settings: Settings,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not settings.upstream_api_key:
            raise ValueError(
                "UPSTREAM_MODE=remote requires UPSTREAM_API_KEY to be configured."
            )
        parsed = urlsplit(settings.upstream_base_url)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "Remote upstream base URLs must be HTTPS URLs without "
                "credentials, query strings, or fragments."
            )
        path = parsed.path.rstrip("/")
        if not path.endswith("/chat/completions"):
            path = f"{path}/chat/completions"
        self.url = urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))
        self.api_key = settings.upstream_api_key
        self.max_response_bytes = settings.max_upstream_response_bytes
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.upstream_timeout_seconds),
            follow_redirects=False,
            transport=transport,
        )

    async def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = await self.client.post(
                self.url,
                headers=self._headers(),
                json=payload,
            )
            response.raise_for_status()
            if len(response.content) > self.max_response_bytes:
                raise UpstreamError(
                    "The configured model upstream response exceeded the size limit."
                )
            data = response.json()
        except httpx.HTTPStatusError as error:
            raise UpstreamError(
                "The configured model upstream rejected the request.",
                upstream_status_code=error.response.status_code,
            ) from error
        except (httpx.HTTPError, ValueError) as error:
            raise UpstreamError(
                "The configured model upstream did not return a valid response."
            ) from error
        if not isinstance(data, dict):
            raise UpstreamError(
                "The configured model upstream returned an invalid response."
            )
        return data

    async def complete_stream(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Buffer and parse SSE so output can be screened before client delivery."""
        state: dict[str, Any] = {
            "id": None,
            "created": None,
            "model": payload.get("model", ""),
            "system_fingerprint": None,
            "choices": {},
            "usage": None,
        }
        pending_data: list[str] = []
        total_bytes = 0
        try:
            async with self.client.stream(
                "POST",
                self.url,
                headers=self._headers(),
                json=payload,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    total_bytes += len(line.encode("utf-8")) + 1
                    if total_bytes > self.max_response_bytes:
                        raise UpstreamError(
                            "The configured model stream exceeded the size limit."
                        )
                    if line == "":
                        if _process_sse_data(pending_data, state):
                            break
                        pending_data.clear()
                    elif line.startswith("data:"):
                        pending_data.append(line[5:].lstrip())
                else:
                    _process_sse_data(pending_data, state)
        except UpstreamError:
            raise
        except httpx.HTTPStatusError as error:
            raise UpstreamError(
                "The configured model upstream rejected the stream request.",
                upstream_status_code=error.response.status_code,
            ) from error
        except (httpx.HTTPError, ValueError, json.JSONDecodeError) as error:
            raise UpstreamError(
                "The configured model upstream did not return a valid stream."
            ) from error

        choices_by_index = state["choices"]
        if not choices_by_index:
            raise UpstreamError(
                "The configured model upstream stream contained no choices."
            )
        choices = []
        for index in sorted(choices_by_index):
            choice = choices_by_index[index]
            message = choice["message"]
            for tool_call in message.get("_tool_calls_by_index", {}).values():
                tool_call.pop("_index", None)
            tool_calls = message.pop("_tool_calls_by_index", None)
            if tool_calls:
                message["tool_calls"] = [
                    tool_calls[key] for key in sorted(tool_calls)
                ]
            choices.append(
                {
                    "index": index,
                    "message": message,
                    "finish_reason": choice.get("finish_reason"),
                }
            )

        result: dict[str, Any] = {
            "id": state["id"] or "chatcmpl-buffered",
            "object": "chat.completion",
            "created": state["created"] or 0,
            "model": state["model"],
            "choices": choices,
        }
        if state["usage"] is not None:
            result["usage"] = state["usage"]
        if state["system_fingerprint"] is not None:
            result["system_fingerprint"] = state["system_fingerprint"]
        return result

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "text/event-stream, application/json",
        }

    async def close(self) -> None:
        await self.client.aclose()


def _process_sse_data(
    data_lines: list[str],
    state: dict[str, Any],
) -> bool:
    if not data_lines:
        return False
    data = "\n".join(data_lines)
    if data == "[DONE]":
        return True
    try:
        event = json.loads(data)
    except json.JSONDecodeError as error:
        raise UpstreamError(
            "The configured model upstream returned malformed SSE data."
        ) from error
    if not isinstance(event, dict):
        raise UpstreamError(
            "The configured model upstream returned an invalid SSE event."
        )
    if event.get("error"):
        raise UpstreamError("The configured model upstream reported a stream error.")

    for field in ("id", "created", "model", "system_fingerprint"):
        if event.get(field) is not None:
            state[field] = event[field]
    if event.get("usage") is not None:
        state["usage"] = event["usage"]

    chunks = event.get("choices", [])
    if not isinstance(chunks, list):
        raise UpstreamError(
            "The configured model upstream returned invalid stream choices."
        )
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        index = chunk.get("index", 0)
        if not isinstance(index, int) or index < 0:
            raise UpstreamError(
                "The configured model upstream returned an invalid choice index."
            )
        choice = state["choices"].setdefault(
            index,
            {"message": {"role": "assistant"}, "finish_reason": None},
        )
        delta = chunk.get("delta") or {}
        if isinstance(delta, dict):
            _merge_message_delta(choice["message"], delta)
        if chunk.get("finish_reason") is not None:
            choice["finish_reason"] = chunk["finish_reason"]
    return False


def _merge_message_delta(message: dict[str, Any], delta: dict[str, Any]) -> None:
    for field, value in delta.items():
        if field == "tool_calls" and isinstance(value, list):
            tool_calls = message.setdefault("_tool_calls_by_index", {})
            for item in value:
                if not isinstance(item, dict):
                    continue
                index = item.get("index", 0)
                if not isinstance(index, int) or index < 0:
                    continue
                tool = tool_calls.setdefault(index, {})
                for key in ("id", "type"):
                    if item.get(key) is not None:
                        tool.setdefault(key, item[key])
                function_delta = item.get("function")
                if isinstance(function_delta, dict):
                    function = tool.setdefault("function", {})
                    for key, text in function_delta.items():
                        if isinstance(text, str):
                            function[key] = function.get(key, "") + text
                        else:
                            function[key] = text
                continue

        if isinstance(value, str):
            if field == "role" or field not in message:
                message[field] = value
            elif isinstance(message[field], str):
                message[field] += value
            else:
                message[field] = value
        elif isinstance(value, list) and isinstance(message.get(field), list):
            message[field].extend(value)
        else:
            message[field] = value


def create_upstream(settings: Settings) -> Upstream:
    if settings.upstream_mode == "demo":
        return DemoUpstream()
    return OpenAICompatibleUpstream(settings)