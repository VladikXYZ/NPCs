from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from .schemas import fingerprint


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def git_state(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=False
        ).stdout.strip()

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {"commit": commit or None, "dirty": bool(status), "status": status.splitlines()}


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment_provenance(root: Path) -> dict[str, Any]:
    return {
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "packages": {
            "llama-cpp-python": package_version("llama-cpp-python"),
            "Jinja2": package_version("Jinja2"),
        },
        "git": git_state(root),
    }


def implementation_fingerprint(root: Path) -> str:
    paths = [
        root / "npc_runtime.py", root / "chat_templates.py", root / "utils.py",
        *sorted((root / "npcbench").glob("*.py")),
    ]
    return fingerprint({str(path.relative_to(root)): sha256_file(path) for path in paths})


def model_provenance(model: dict[str, Any], *, hash_weights: bool = True) -> dict[str, Any]:
    path = Path(model["path"]).resolve()
    stat = path.stat()
    actual_sha256 = sha256_file(path) if hash_weights else None
    declared_sha256 = model.get("sha256")
    if actual_sha256 and declared_sha256 and actual_sha256.casefold() != str(declared_sha256).casefold():
        raise ValueError(f"Model checksum mismatch for {model['name']}: registry does not match {path.name}")
    return {
        "model_id": model_id(model),
        "name": model["name"],
        "family": model.get("family"),
        "path": str(path),
        "size_bytes": stat.st_size,
        "sha256": actual_sha256,
        "declared_sha256": declared_sha256,
        "registry": {key: value for key, value in model.items() if key != "path"},
    }


def model_id(model: dict[str, Any]) -> str:
    if isinstance(model.get("id"), str) and model["id"].strip():
        return model["id"].strip()
    slug = "-".join("".join(ch.casefold() if ch.isalnum() else " " for ch in model["name"]).split())
    suffix = hashlib.sha256(str(model["path"]).encode("utf-8")).hexdigest()[:8]
    return f"{slug}-{suffix}"
