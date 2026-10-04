"""Fast deterministic rules for direct and lightly obfuscated prompt attacks."""

from __future__ import annotations

import base64
import binascii
import re
import unicodedata
from dataclasses import dataclass


ATTACK_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "instruction_override",
        re.compile(
            r"\bignore\s+(?:all\s+)?(?:previous|prior|above|earlier)\s+"
            r"(?:instructions?|rules?|messages?|prompts?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "hidden_prompt_disclosure",
        re.compile(
            r"\b(?:reveal|print|show|repeat|dump|expose|leak)\s+(?:the\s+)?"
            r"(?:hidden|system|developer|initial|secret)\s+"
            r"(?:prompt|instructions?|message|context)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "jailbreak_persona",
        re.compile(
            r"\b(?:you\s+are\s+now|act\s+as|pretend\s+to\s+be)\s+"
            r"(?:an?\s+)?(?:dan|jailbroken|uncensored|unrestricted)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "privileged_mode",
        re.compile(
            r"\b(?:enable|activate|enter|switch\s+to)\s+"
            r"(?:developer|god|dan|jailbreak)\s+mode\b",
            re.IGNORECASE,
        ),
    ),
    (
        "credential_exfiltration",
        re.compile(
            r"\b(?:send|upload|forward|post|exfiltrate|email|transmit)\b.{0,80}"
            r"\b(?:api\s+keys?|passwords?|credentials?|access\s+tokens?|"
            r"secrets?|private\s+keys?)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "override_marker",
        re.compile(
            r"(?:\[\s*(?:system|developer)\s*(?:prompt|instructions?)?\s*\]"
            r"|<\s*/?\s*(?:system|developer)\s*>)",
            re.IGNORECASE,
        ),
    ),
)

_BASE64_CANDIDATE = re.compile(
    r"(?<![A-Za-z0-9_+/-])[A-Za-z0-9_+/-]{24,}={0,2}(?![A-Za-z0-9_+/-])"
)
_UNICODE_ESCAPE = re.compile(r"\\u([0-9a-fA-F]{4})")


@dataclass(frozen=True, slots=True)
class RuleScan:
    """Rule results without retaining or returning the original prompt."""

    findings: tuple[str, ...]

    @property
    def blocked(self) -> bool:
        return bool(self.findings)


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    return "".join(
        character
        for character in normalized
        if unicodedata.category(character) not in {"Cf", "Cc"}
        or character in "\n\t"
    )


def _decode_unicode_escapes(text: str) -> str | None:
    if not _UNICODE_ESCAPE.search(text):
        return None
    return _UNICODE_ESCAPE.sub(
        lambda match: chr(int(match.group(1), 16)),
        text,
    )


def _decoded_base64_candidates(text: str) -> list[str]:
    decoded: list[str] = []
    for match in _BASE64_CANDIDATE.finditer(text):
        candidate = match.group(0)
        if len(candidate) > 4096:
            continue
        try:
            raw = base64.b64decode(
                candidate + "=" * (-len(candidate) % 4),
                altchars=b"-_",
                validate=True,
            )
            value = raw.decode("utf-8")
        except (binascii.Error, UnicodeDecodeError, ValueError):
            continue
        if value and all(
            character.isprintable() or character in "\r\n\t" for character in value
        ):
            decoded.append(value)
    return decoded


class HeuristicDetector:
    """Detect high-confidence instruction attacks and simple encoding smuggling."""

    def scan(self, text: str) -> RuleScan:
        findings: set[str] = set()
        variants = [_normalize(text)]

        escaped = _decode_unicode_escapes(text)
        if escaped:
            variants.append(_normalize(escaped))

        for variant in tuple(variants):
            variants.extend(
                _normalize(candidate)
                for candidate in _decoded_base64_candidates(variant)
            )

        for index, variant in enumerate(variants):
            for rule_name, pattern in ATTACK_RULES:
                if pattern.search(variant):
                    original_variants = {_normalize(text), _normalize(escaped or "")}
                    if index > 0 and variant not in original_variants:
                        findings.add(f"encoded:{rule_name}")
                    elif index > 0:
                        findings.add(f"obfuscated:{rule_name}")
                    else:
                        findings.add(rule_name)

        return RuleScan(findings=tuple(sorted(findings)))