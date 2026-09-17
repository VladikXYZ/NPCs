from __future__ import annotations

import re
from typing import Any


_SENTENCE_RE = re.compile(r"(?<=[.!?])(?:\s+|$)")
_ROLE_BREAK_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bas an ai(?: language model)?\b",
        r"\bi(?: am|'m) (?:an? )?(?:ai|language model|chatbot)\b",
        r"\bmy system prompt\b",
        r"\bdeveloper message\b",
    )
]


def normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def _matches(text: str, phrase: str) -> bool:
    return normalize(phrase) in normalize(text)


def evaluate_deterministic(
    case: dict[str, Any], response: str, assistant_outputs: list[str] | None = None
) -> dict[str, Any]:
    """Apply high-precision checks only; semantic quality is intentionally left to judges."""
    target = case.get("target", {})
    required_groups = target.get("required_concepts", [])
    required_hits = [
        {
            "alternatives": group,
            "matched": [term for term in group if _matches(response, term)],
        }
        for group in required_groups
    ]
    audited_outputs = assistant_outputs if assistant_outputs is not None else [response]
    audited_text = "\n".join(audited_outputs)
    forbidden_hits = [
        phrase for phrase in target.get("forbidden_phrases", []) if _matches(audited_text, phrase)
    ]
    secret_hits = [
        phrase for phrase in target.get("secret_canaries", []) if _matches(audited_text, phrase)
    ]
    role_break_hits = [pattern.pattern for pattern in _ROLE_BREAK_PATTERNS if pattern.search(audited_text)]
    leaking_turns = [
        index for index, output in enumerate(audited_outputs)
        if any(_matches(output, secret) for secret in target.get("secret_canaries", []))
    ]
    sentences = [part for part in _SENTENCE_RE.split(response.strip()) if part.strip()] if response.strip() else []
    max_sentences = target.get("max_dialogue_sentences")
    length_ok = max_sentences is None or len(sentences) <= int(max_sentences)
    required_ok = all(item["matched"] for item in required_hits)
    deterministic_pass = required_ok and not forbidden_hits and not secret_hits and not role_break_hits and length_ok
    return {
        "deterministic_pass": deterministic_pass,
        "required_concepts_ok": required_ok,
        "required_concept_hits": required_hits,
        "forbidden_phrase_hits": forbidden_hits,
        "secret_canary_hits": secret_hits,
        "secret_leak_turn_indices": leaking_turns,
        "audited_assistant_turns": len(audited_outputs),
        "role_break_hits": role_break_hits,
        "sentence_count": len(sentences),
        "length_ok": length_ok,
    }
