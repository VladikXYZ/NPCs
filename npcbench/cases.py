from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schemas import fingerprint, validate_cases


def load_cases(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    if source.suffix.lower() == ".jsonl":
        cases = [
            json.loads(line)
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        payload = json.loads(source.read_text(encoding="utf-8"))
        cases = payload["cases"] if isinstance(payload, dict) and "cases" in payload else payload
    if not isinstance(cases, list):
        raise ValueError(f"{source}: expected a JSON array or JSONL records")
    return validate_cases(cases)


def dataset_fingerprint(cases: list[dict[str, Any]]) -> str:
    return fingerprint(cases)


def select_cases(
    cases: list[dict[str, Any]],
    suites: set[str] | None = None,
    case_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    return [
        case
        for case in cases
        if (not suites or case["suite"] in suites)
        and (not case_ids or case["case_id"] in case_ids)
    ]

