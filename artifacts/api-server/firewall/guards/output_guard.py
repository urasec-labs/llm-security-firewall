"""Output scanning and redaction for secrets, common PII, and URLs."""

from __future__ import annotations

import re
from dataclasses import dataclass


_PRIVATE_KEY = re.compile(
    r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"
    r"[\s\S]*?-----END (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----",
    re.IGNORECASE,
)
_SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "api_key",
        re.compile(
            r"\b(?:sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|"
            r"github_pat_[A-Za-z0-9_]{20,}|glpat-[A-Za-z0-9_-]{20,}|"
            r"xox[baprs]-[A-Za-z0-9-]{12,}|AKIA[0-9A-Z]{16})\b"
        ),
    ),
    (
        "credential_assignment",
        re.compile(
            r"(?i)\b(?:api[_ -]?key|access[_ -]?token|secret[_ -]?key|"
            r"password|client[_ -]?secret)\b\s*[:=]\s*"
            r"[\"']?[A-Za-z0-9_./+=:-]{8,}"
        ),
    ),
    (
        "bearer_token",
        re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{12,}"),
    ),
    (
        "jwt",
        re.compile(
            r"(?<![A-Za-z0-9_-])eyJ[A-Za-z0-9_-]{8,}\."
            r"[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"
        ),
    ),
    (
        "email",
        re.compile(
            r"\b[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@"
            r"[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+\b"
        ),
    ),
    (
        "ssn",
        re.compile(r"(?<!\d)(?:\d{3}-\d{2}-\d{4})(?!\d)"),
    ),
    (
        "phone",
        re.compile(r"(?<!\w)\+?\d(?:[\d .()\-]{7,}\d)(?!\w)"),
    ),
    (
        "url",
        re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE),
    ),
)
_CARD_CANDIDATE = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")


@dataclass(frozen=True, slots=True)
class RedactionResult:
    text: str
    findings: tuple[str, ...]

    @property
    def redacted(self) -> bool:
        return bool(self.findings)


def _passes_luhn(value: str) -> bool:
    digits = [int(character) for character in value if character.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    checksum = 0
    parity = len(digits) % 2
    for index, digit in enumerate(digits):
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0


class OutputGuard:
    """Redact sensitive output instead of returning the original value."""

    def __init__(self, redact_urls: bool = True) -> None:
        self.redact_urls = redact_urls

    def redact(self, text: str) -> RedactionResult:
        findings: set[str] = set()
        safe_text = text

        def redact_private_key(match: re.Match[str]) -> str:
            findings.add("private_key")
            return "[REDACTED_PRIVATE_KEY]"

        safe_text = _PRIVATE_KEY.sub(redact_private_key, safe_text)

        def redact_card(match: re.Match[str]) -> str:
            if _passes_luhn(match.group(0)):
                findings.add("payment_card")
                return "[REDACTED_PAYMENT_CARD]"
            return match.group(0)

        safe_text = _CARD_CANDIDATE.sub(redact_card, safe_text)

        for finding, pattern in _SECRET_PATTERNS:
            if finding == "url" and not self.redact_urls:
                continue

            def replacement(_match: re.Match[str], label: str = finding) -> str:
                findings.add(label)
                return f"[REDACTED_{label.upper()}]"

            safe_text = pattern.sub(replacement, safe_text)

        return RedactionResult(
            text=safe_text,
            findings=tuple(sorted(findings)),
        )