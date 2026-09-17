from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable


SCHEMA_VERSION = "0.1"
SUITES = {
    "identity_style",
    "known_lore",
    "unknown_lore",
    "conflicting_information",
    "state_tracking",
    "memory",
    "robustness",
    "jailbreak",
    "performance",
}
ROLES = {"user", "assistant"}
STATUSES = {"ok", "timeout", "error", "invalid_output", "skipped"}


class SchemaError(ValueError):
    pass


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _require(mapping: dict[str, Any], key: str, expected: type, where: str) -> Any:
    if key not in mapping:
        raise SchemaError(f"{where}: missing required field {key!r}")
    value = mapping[key]
    if not isinstance(value, expected):
        raise SchemaError(f"{where}.{key}: expected {expected.__name__}, got {type(value).__name__}")
    return value


def _nonempty_string(mapping: dict[str, Any], key: str, where: str) -> str:
    value = _require(mapping, key, str, where).strip()
    if not value:
        raise SchemaError(f"{where}.{key}: must not be empty")
    return value


def validate_case(case: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(case, dict):
        raise SchemaError("case: expected object")
    if case.get("schema_version") != SCHEMA_VERSION:
        raise SchemaError(f"case.schema_version: expected {SCHEMA_VERSION!r}")
    _nonempty_string(case, "case_id", "case")
    _nonempty_string(case, "world_id", "case")
    suite = _nonempty_string(case, "suite", "case")
    if suite not in SUITES:
        raise SchemaError(f"case.suite: unsupported value {suite!r}")
    if case.get("split") not in {"dev", "test"}:
        raise SchemaError("case.split: expected 'dev' or 'test'")

    npc = _require(case, "npc", dict, "case")
    for key in ("npc_id", "name", "persona", "speaking_style"):
        _nonempty_string(npc, key, "case.npc")

    messages = _require(case, "messages", list, "case")
    if not messages:
        raise SchemaError("case.messages: must contain at least one message")
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise SchemaError(f"case.messages[{index}]: expected object")
        if message.get("role") not in ROLES:
            raise SchemaError(f"case.messages[{index}].role: invalid role")
        _nonempty_string(message, "content", f"case.messages[{index}]")
    if messages[-1]["role"] != "user":
        raise SchemaError("case.messages: final message must have role 'user'")

    state = _require(case, "initial_state", dict, "case")
    for key in ("public_facts", "private_facts", "allowed_actions"):
        values = _require(state, key, list, "case.initial_state")
        if not all(isinstance(value, str) and value.strip() for value in values):
            raise SchemaError(f"case.initial_state.{key}: expected non-empty strings")

    target = _require(case, "target", dict, "case")
    _nonempty_string(target, "description", "case.target")
    required = target.get("required_concepts", [])
    if not isinstance(required, list) or not all(
        isinstance(group, list)
        and group
        and all(isinstance(term, str) and term.strip() for term in group)
        for group in required
    ):
        raise SchemaError("case.target.required_concepts: expected list of non-empty synonym lists")
    for key in ("forbidden_phrases", "secret_canaries"):
        values = target.get(key, [])
        if not isinstance(values, list) or not all(isinstance(v, str) and v.strip() for v in values):
            raise SchemaError(f"case.target.{key}: expected non-empty strings")

    metadata = _require(case, "metadata", dict, "case")
    _nonempty_string(metadata, "source", "case.metadata")
    return case


def validate_cases(cases: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    validated: list[dict[str, Any]] = []
    ids: set[str] = set()
    for case in cases:
        validate_case(case)
        case_id = case["case_id"]
        if case_id in ids:
            raise SchemaError(f"duplicate case_id: {case_id}")
        ids.add(case_id)
        validated.append(case)
    return validated


def validate_result(record: dict[str, Any]) -> dict[str, Any]:
    if record.get("schema_version") != SCHEMA_VERSION:
        raise SchemaError("result.schema_version: unsupported version")
    for key in ("run_id", "case_id", "model_id", "mode", "status"):
        _nonempty_string(record, key, "result")
    if record["status"] not in STATUSES:
        raise SchemaError(f"result.status: invalid value {record['status']!r}")
    if not isinstance(record.get("attempt"), int) or record["attempt"] < 0:
        raise SchemaError("result.attempt: expected non-negative integer")
    if record["status"] == "ok" and not isinstance(record.get("response"), str):
        raise SchemaError("result.response: successful result requires response text")
    return record


def result_key(record: dict[str, Any]) -> tuple[str, str, str, int]:
    return (
        str(record["case_id"]),
        str(record["model_id"]),
        str(record["mode"]),
        int(record["attempt"]),
    )


@dataclass(frozen=True)
class RunKey:
    case_id: str
    model_id: str
    mode: str
    attempt: int

    def as_tuple(self) -> tuple[str, str, str, int]:
        return self.case_id, self.model_id, self.mode, self.attempt
