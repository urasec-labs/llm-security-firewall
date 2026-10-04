"""Client authentication and bounded in-memory request rate limiting."""

from __future__ import annotations

import hashlib
import hmac
import math
import time
from collections import deque
from threading import Lock

from fastapi import HTTPException, Request

from firewall.config import Settings


class InMemoryRateLimiter:
    """A process-local sliding-window limiter with bounded key storage."""

    def __init__(self, settings: Settings) -> None:
        self.limit = settings.rate_limit_requests
        self.window_seconds = settings.rate_limit_window_seconds
        self.max_keys = settings.rate_limit_max_keys
        self._buckets: dict[str, deque[float]] = {}
        self._lock = Lock()

    def consume(self, key: str) -> tuple[bool, int]:
        now = time.monotonic()
        cutoff = now - self.window_seconds
        with self._lock:
            if key not in self._buckets and len(self._buckets) >= self.max_keys:
                self._prune_all(cutoff)
                if len(self._buckets) >= self.max_keys:
                    return False, math.ceil(self.window_seconds)

            bucket = self._buckets.setdefault(key, deque())
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()
            if len(bucket) >= self.limit:
                retry_after = max(
                    1,
                    math.ceil(bucket[0] + self.window_seconds - now),
                )
                return False, retry_after
            bucket.append(now)
            return True, 0

    def _prune_all(self, cutoff: float) -> None:
        expired = []
        for key, bucket in self._buckets.items():
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()
            if not bucket:
                expired.append(key)
        for key in expired:
            del self._buckets[key]


def request_identity(request: Request, bearer_token: str | None) -> str:
    """Return a stable, non-secret rate-limit identity."""
    if bearer_token:
        subject = f"token:{bearer_token}"
    else:
        host = request.client.host if request.client else "unknown"
        subject = f"ip:{host}"
    return hashlib.sha256(subject.encode("utf-8")).hexdigest()[:20]


def enforce_access(
    request: Request,
    settings: Settings,
    limiter: InMemoryRateLimiter,
) -> None:
    """Require a configured bearer key and consume the matching rate bucket."""
    authorization = request.headers.get("authorization", "")
    scheme, _, supplied_key = authorization.partition(" ")
    supplied_key = supplied_key.strip()
    valid_bearer = scheme.lower() == "bearer" and bool(supplied_key)

    if settings.auth_required and not settings.client_api_key:
        request.state.audit_event = "auth_not_configured"
        raise HTTPException(
            status_code=503,
            detail="Client authentication is required but not configured.",
        )

    if settings.auth_required:
        valid_bearer = bool(
            valid_bearer
            and hmac.compare_digest(
                supplied_key,
                settings.client_api_key or "",
            )
        )
        if not valid_bearer:
            request.state.audit_client_id = request_identity(request, None)
            allowed, retry_after = limiter.consume(request.state.audit_client_id)
            if not allowed:
                request.state.audit_event = "rate_limited"
                raise HTTPException(
                    status_code=429,
                    detail="Request rate limit exceeded.",
                    headers={"Retry-After": str(retry_after)},
                )
            request.state.audit_event = "auth_failed"
            raise HTTPException(
                status_code=401,
                detail="A valid bearer token is required.",
                headers={"WWW-Authenticate": "Bearer"},
            )

    identity = request_identity(
        request,
        supplied_key if valid_bearer else None,
    )
    request.state.audit_client_id = identity
    allowed, retry_after = limiter.consume(identity)
    if not allowed:
        request.state.audit_event = "rate_limited"
        raise HTTPException(
            status_code=429,
            detail="Request rate limit exceeded.",
            headers={"Retry-After": str(retry_after)},
        )