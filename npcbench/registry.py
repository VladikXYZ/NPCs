from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


RELEASE_FIELDS = ("id", "source", "revision", "quantization", "parameter_count_b", "sha256", "template_status")


def audit_registry(path: str | Path, root: str | Path) -> dict[str, Any]:
    source = Path(path)
    root_path = Path(root).resolve()
    try:
        display_source = source.resolve().relative_to(root_path)
    except ValueError:
        display_source = source
    records = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("Model registry must be a JSON array.")
    names = Counter(record.get("name") for record in records)
    paths = Counter(record.get("path") for record in records)
    findings = []
    for index, record in enumerate(records):
        model_path = (root_path / str(record.get("path", ""))).resolve()
        installed = model_path.is_file()
        eligible = installed and not record.get("hidden", False)
        missing = [field for field in RELEASE_FIELDS if record.get(field) in (None, "")]
        if record.get("params") == 122:
            findings.append({"index": index, "name": record.get("name"), "severity": "error", "issue": "placeholder params=122 must not be used"})
        if missing and eligible:
            findings.append({"index": index, "name": record.get("name"), "severity": "error", "issue": "missing release metadata", "fields": missing})
        if eligible and record.get("sha256") and not re.fullmatch(r"[0-9a-fA-F]{64}", str(record["sha256"])):
            findings.append({"index": index, "name": record.get("name"), "severity": "error", "issue": "invalid SHA-256 format"})
        if eligible and record.get("template_sha256") and not re.fullmatch(r"[0-9a-fA-F]{64}", str(record["template_sha256"])):
            findings.append({"index": index, "name": record.get("name"), "severity": "error", "issue": "invalid template SHA-256 format"})
        if not installed:
            findings.append({"index": index, "name": record.get("name"), "severity": "info", "issue": "weight not installed", "path": str(record.get("path", ""))})
    duplicates = {
        "names": sorted(name for name, count in names.items() if name and count > 1),
        "paths": sorted(path for path, count in paths.items() if path and count > 1),
    }
    if duplicates["names"] or duplicates["paths"]:
        findings.append({"severity": "error", "issue": "duplicate registry identity", **duplicates})
    errors = sum(finding["severity"] == "error" for finding in findings)
    return {
        "registry": display_source.as_posix(), "models": len(records),
        "installed_models": sum((root_path / str(record.get("path", ""))).is_file() for record in records),
        "eligible_installed_models": sum(
            (root_path / str(record.get("path", ""))).is_file() and not record.get("hidden", False)
            for record in records
        ),
        "release_ready": errors == 0,
        "error_count": errors, "duplicates": duplicates, "findings": findings,
    }
