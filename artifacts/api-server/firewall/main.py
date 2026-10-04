"""FastAPI entry point for the LLM security gateway."""

from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from typing import Any, Iterator

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from firewall.access_control import (
    InMemoryRateLimiter,
    enforce_access,
    request_identity,
)
from firewall.audit import get_audit_logger
from firewall.config import Settings
from firewall.gateway import Gateway, decision_to_dict
from firewall.models import (
    ChatCompletionRequest,
    HealthResponse,
    ScanResponse,
    TextScanRequest,
)
from firewall.upstream import UpstreamError


class RequestTooLarge(Exception):
    """Raised when the streaming ASGI body exceeds the configured limit."""


class BodyLimitMiddleware:
    """Enforce a byte limit even when a client omits Content-Length."""

    def __init__(self, app: ASGIApp, max_body_bytes: int) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        content_length = headers.get(b"content-length")
        if content_length:
            try:
                if int(content_length) > self.max_body_bytes:
                    await self._send_too_large(scope, receive, send)
                    return
            except ValueError:
                pass

        received_bytes = 0

        async def limited_receive() -> Message:
            nonlocal received_bytes
            message = await receive()
            if message["type"] == "http.request":
                received_bytes += len(message.get("body", b""))
                if received_bytes > self.max_body_bytes:
                    raise RequestTooLarge
            return message

        try:
            await self.app(scope, limited_receive, send)
        except RequestTooLarge:
            await self._send_too_large(scope, receive, send)

    @staticmethod
    async def _send_too_large(scope: Scope, receive: Receive, send: Send) -> None:
        response = JSONResponse(
            status_code=413,
            content={
                "error": {
                    "type": "request_too_large",
                    "message": "Request body exceeds the configured size limit.",
                }
            },
        )
        await response(scope, receive, send)


def _create_app(settings: Settings | None = None) -> FastAPI:
    runtime_settings = settings or Settings.from_env()
    limiter = InMemoryRateLimiter(runtime_settings)
    audit_logger = get_audit_logger()
    bearer_scheme = HTTPBearer(auto_error=False)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        gateway = Gateway(runtime_settings)
        await gateway.initialize()
        app.state.gateway = gateway
        try:
            yield
        finally:
            await gateway.close()

    app = FastAPI(
        title="LLM Security Gateway",
        description=(
            "OpenAI-compatible chat proxy with input screening and output redaction."
        ),
        version="0.1.0",
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        redoc_url=None,
        lifespan=lifespan,
    )
    app.add_middleware(
        BodyLimitMiddleware,
        max_body_bytes=runtime_settings.max_body_bytes,
    )

    @app.middleware("http")
    async def audit_requests(request: Request, call_next: Any) -> Any:
        started = time.perf_counter()
        request.state.audit_client_id = request_identity(request, None)
        request.state.audit_event = "request_complete"
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            if status_code >= 400 and request.state.audit_event == "request_complete":
                request.state.audit_event = "http_error"
            return response
        finally:
            audit_record: dict[str, Any] = {
                "event": request.state.audit_event,
                "method": request.method,
                "path": request.url.path,
                "status_code": status_code,
                "duration_ms": round(
                    (time.perf_counter() - started) * 1000,
                    3,
                ),
                "client_id": request.state.audit_client_id,
                "findings": getattr(request.state, "audit_findings", []),
            }
            upstream_status = getattr(
                request.state,
                "upstream_status_code",
                None,
            )
            if upstream_status is not None:
                audit_record["upstream_status_code"] = upstream_status
            audit_logger.info(
                "request",
                extra={"audit_record": audit_record},
            )

    async def access_guard(
        request: Request,
        _credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    ) -> None:
        enforce_access(request, runtime_settings, limiter)

    @app.get("/api/healthz", response_model=HealthResponse)
    async def healthz(request: Request) -> HealthResponse:
        gateway: Gateway | None = getattr(request.app.state, "gateway", None)
        return HealthResponse(
            status="ok",
            mode=runtime_settings.upstream_mode,
            semantic_enabled=runtime_settings.semantic_enabled,
            classifier_ready=bool(gateway and gateway.input_guard.classifier.ready),
        )

    @app.get("/api/metrics", dependencies=[Depends(access_guard)])
    async def metrics(request: Request) -> dict[str, int | float]:
        return request.app.state.gateway.metrics.snapshot()

    @app.post(
        "/api/scan/input",
        response_model=ScanResponse,
        dependencies=[Depends(access_guard)],
    )
    async def scan_input(
        body: TextScanRequest,
        request: Request,
    ) -> dict[str, Any]:
        if len(body.text) > runtime_settings.max_prompt_chars:
            raise HTTPException(
                status_code=413,
                detail="Prompt exceeds the configured character limit.",
            )
        started = time.perf_counter()
        decision = await request.app.state.gateway.input_guard.scan(body.text)
        request.app.state.gateway.metrics.record_guard(
            (time.perf_counter() - started) * 1000
        )
        request.state.audit_findings = list(decision.findings)
        if decision.blocked:
            request.state.audit_event = "input_blocked"
        return decision_to_dict(decision)

    @app.post("/api/scan/output", dependencies=[Depends(access_guard)])
    async def scan_output(
        body: TextScanRequest,
        request: Request,
    ) -> dict[str, Any]:
        result = request.app.state.gateway.output_guard.redact(body.text)
        if result.redacted:
            request.app.state.gateway.metrics.record_redaction()
            request.state.audit_event = "output_redacted"
            request.state.audit_findings = list(result.findings)
        return {
            "text": result.text,
            "redacted": result.redacted,
            "findings": list(result.findings),
        }

    @app.post(
        "/api/chat/completions",
        response_model=None,
        dependencies=[Depends(access_guard)],
    )
    async def chat_completions(
        body: ChatCompletionRequest,
        request: Request,
    ) -> JSONResponse | StreamingResponse | dict[str, Any]:
        gateway: Gateway = request.app.state.gateway
        gateway.metrics.record_chat()

        untrusted_messages = [
            message.content
            for message in body.messages
            if message.role in {"user", "tool"}
        ]
        if not untrusted_messages:
            raise HTTPException(
                status_code=400,
                detail="At least one user or tool message is required.",
            )

        prompt = "\n".join(untrusted_messages)
        if len(prompt) > runtime_settings.max_prompt_chars:
            raise HTTPException(
                status_code=413,
                detail="Prompt exceeds the configured character limit.",
            )
        started = time.perf_counter()
        decision = await gateway.input_guard.scan(prompt)
        gateway.metrics.record_guard((time.perf_counter() - started) * 1000)

        if decision.blocked:
            gateway.metrics.record_blocked()
            request.state.audit_event = "input_blocked"
            request.state.audit_findings = list(decision.findings)
            return JSONResponse(
                status_code=403,
                content={
                    "error": {
                        "type": "policy_violation",
                        "code": "unsafe_prompt",
                        "message": "Prompt blocked by input security policy.",
                        "findings": list(decision.findings),
                        "risk_score": round(decision.risk_score, 4),
                    }
                },
            )

        payload = body.model_dump(exclude_none=True)
        try:
            safe_response, findings = await gateway.complete(payload)
        except UpstreamError as error:
            request.state.audit_event = "upstream_error"
            request.state.upstream_status_code = error.upstream_status_code
            raise HTTPException(
                status_code=502,
                detail="The configured model upstream could not complete the request.",
            ) from error

        if findings:
            request.state.audit_event = "output_redacted"
            request.state.audit_findings = findings
            safe_response["security"] = {
                "output_redacted": True,
                "findings": findings,
            }
        if body.stream:
            return StreamingResponse(
                _as_openai_sse(safe_response),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                    "X-LLM-Security-Redacted": str(bool(findings)).lower(),
                    "X-LLM-Security-Findings": ",".join(findings),
                },
            )
        return safe_response

    return app


def _as_openai_sse(response: dict[str, Any]) -> Iterator[bytes]:
    """Emit a sanitized completion as OpenAI-compatible SSE frames."""
    metadata: dict[str, Any] = {
        "id": response.get("id", "chatcmpl-buffered"),
        "object": "chat.completion.chunk",
        "created": response.get("created", 0),
        "model": response.get("model", ""),
    }
    if response.get("system_fingerprint") is not None:
        metadata["system_fingerprint"] = response["system_fingerprint"]

    for choice in response.get("choices", []):
        if not isinstance(choice, dict):
            continue
        index = choice.get("index", 0)
        message = choice.get("message", {})
        if not isinstance(message, dict):
            message = {}
        yield _sse_frame(
            {
                **metadata,
                "choices": [
                    {"index": index, "delta": message, "finish_reason": None}
                ],
            }
        )
        yield _sse_frame(
            {
                **metadata,
                "choices": [
                    {
                        "index": index,
                        "delta": {},
                        "finish_reason": choice.get("finish_reason") or "stop",
                    }
                ],
            }
        )

    if isinstance(response.get("usage"), dict):
        yield _sse_frame(
            {
                **metadata,
                "choices": [],
                "usage": response["usage"],
            }
        )
    yield b"data: [DONE]\n\n"


def _sse_frame(payload: dict[str, Any]) -> bytes:
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"data: {encoded}\n\n".encode("utf-8")


app = _create_app()