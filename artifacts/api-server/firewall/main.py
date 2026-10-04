"""FastAPI entry point for the LLM security gateway."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

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

    @app.get("/api/healthz", response_model=HealthResponse)
    async def healthz(request: Request) -> HealthResponse:
        gateway: Gateway | None = getattr(request.app.state, "gateway", None)
        return HealthResponse(
            status="ok",
            mode=runtime_settings.upstream_mode,
            semantic_enabled=runtime_settings.semantic_enabled,
            classifier_ready=bool(gateway and gateway.input_guard.classifier.ready),
        )

    @app.get("/api/metrics")
    async def metrics(request: Request) -> dict[str, int | float]:
        return request.app.state.gateway.metrics.snapshot()

    @app.post("/api/scan/input", response_model=ScanResponse)
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
        return decision_to_dict(decision)

    @app.post("/api/scan/output")
    async def scan_output(
        body: TextScanRequest,
        request: Request,
    ) -> dict[str, Any]:
        result = request.app.state.gateway.output_guard.redact(body.text)
        if result.redacted:
            request.app.state.gateway.metrics.record_redaction()
        return {
            "text": result.text,
            "redacted": result.redacted,
            "findings": list(result.findings),
        }

    @app.post("/api/chat/completions", response_model=None)
    async def chat_completions(
        body: ChatCompletionRequest,
        request: Request,
    ) -> JSONResponse | dict[str, Any]:
        gateway: Gateway = request.app.state.gateway
        gateway.metrics.record_chat()
        if body.stream:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Streaming is not supported until incremental output "
                    "scanning is enabled."
                ),
            )

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
            raise HTTPException(
                status_code=502,
                detail="The configured model upstream could not complete the request.",
            ) from error

        if findings:
            safe_response["security"] = {
                "output_redacted": True,
                "findings": findings,
            }
        return safe_response

    return app


app = _create_app()