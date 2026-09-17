from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

from .schemas import result_key, validate_result


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
    with destination.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: str | Path, *, validate: bool = False) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{source}:{line_number}: invalid JSON: {exc}") from exc
        if validate:
            validate_result(record)
        records.append(record)
    return records


def completed_keys(records: Iterable[dict[str, Any]], *, retry_failures: bool = False) -> set[tuple[str, str, str, int]]:
    return {
        result_key(record)
        for record in records
        if not retry_failures or record.get("status") == "ok"
    }


def assert_unique_results(records: Iterable[dict[str, Any]]) -> set[tuple[str, str, str, int]]:
    keys: set[tuple[str, str, str, int]] = set()
    for record in records:
        key = result_key(record)
        if key in keys:
            raise ValueError(f"Duplicate result key: {key}")
        keys.add(key)
    return keys


def write_json_atomic(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
