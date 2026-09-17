from __future__ import annotations

import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .judge import DIMENSIONS
from .storage import assert_unique_results, read_jsonl, write_json_atomic


def percentile(values: list[float], q: float) -> float | None:
    clean = sorted(value for value in values if math.isfinite(value) and value >= 0)
    if not clean:
        return None
    position = (len(clean) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return clean[low]
    return clean[low] * (high - position) + clean[high] * (position - low)


def bootstrap_mean_ci(values: list[float], *, samples: int = 2000, seed: int = 1729) -> list[float | None]:
    clean = [value for value in values if math.isfinite(value)]
    if not clean:
        return [None, None]
    if len(clean) == 1:
        return [clean[0], clean[0]]
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choice(clean) for _ in clean) for _ in range(samples))
    return [percentile(means, 0.025), percentile(means, 0.975)]


def _latest_judgments(
    rows: Iterable[dict[str, Any]], judge_id: str | None = None
) -> dict[tuple[str, str, str, int], dict[str, Any]]:
    rows = list(rows)
    judge_ids = {row.get("judge_id") for row in rows if row.get("status") == "ok"}
    judge_ids.discard(None)
    if judge_id is None and len(judge_ids) > 1:
        raise ValueError(f"Multiple judges found ({sorted(judge_ids)}); select one with --judge-id.")
    selected = judge_id or (next(iter(judge_ids)) if judge_ids else None)
    latest: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    for row in rows:
        if row.get("status") == "ok" and (selected is None or row.get("judge_id") == selected):
            latest[(row["case_id"], row["model_id"], row["mode"], row["attempt"])] = row
    return latest


def aggregate(
    run_dir: str | Path, *, bootstrap_samples: int = 2000, judge_id: str | None = None
) -> list[dict[str, Any]]:
    directory = Path(run_dir)
    results = read_jsonl(directory / "results.jsonl", validate=True)
    assert_unique_results(results)
    judgments = _latest_judgments(read_jsonl(directory / "judgments.jsonl"), judge_id=judge_id)
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        groups[(row["model_id"], row["mode"], row.get("suite", "unknown"))].append(row)
    summary: list[dict[str, Any]] = []
    for (model, mode, suite), rows in sorted(groups.items()):
        ok = [row for row in rows if row["status"] == "ok"]
        deterministic_values = [
            1.0 if (row.get("deterministic_checks") or {}).get("deterministic_pass") else 0.0
            for row in rows
        ]
        case_scores: dict[str, list[float]] = defaultdict(list)
        judge_scores: dict[str, list[float]] = {dimension: [] for dimension in DIMENSIONS}
        judged = 0
        role_breaks = 0
        secret_leaks = 0
        group_judge_ids: set[str] = set()
        for row in rows:
            case_scores[row["case_id"]].append(
                1.0 if (row.get("deterministic_checks") or {}).get("deterministic_pass") else 0.0
            )
            judgment_row = judgments.get((row["case_id"], row["model_id"], row["mode"], row["attempt"]))
            if judgment_row:
                judged += 1
                group_judge_ids.add(judgment_row["judge_id"])
                judgment = judgment_row["judgment"]
                role_breaks += judgment["role_break"]
                secret_leaks += judgment["secret_leak"]
                for dimension in DIMENSIONS:
                    judge_scores[dimension].append(float(judgment["scores"][dimension]))
        macro_values = [statistics.fmean(values) for values in case_scores.values()]
        exact_secret_leaks = sum(bool((row.get("deterministic_checks") or {}).get("secret_canary_hits")) for row in rows)
        deterministic_classified = sum(row.get("deterministic_checks") is not None for row in rows)
        ttft = [row.get("measurements", {}).get("ttft_seconds", -1) for row in ok]
        total = [row.get("measurements", {}).get("total_seconds", -1) for row in ok]
        tps = [row.get("measurements", {}).get("text_tokens_per_second", -1) for row in ok]
        item: dict[str, Any] = {
            "model_id": model,
            "mode": mode,
            "suite": suite,
            "requests": len(rows),
            "successful": len(ok),
            "coverage": len(ok) / len(rows),
            "status_counts": {status: sum(row["status"] == status for row in rows) for status in sorted({r["status"] for r in rows})},
            "deterministic_micro_rate": statistics.fmean(deterministic_values),
            "deterministic_micro_ci95": bootstrap_mean_ci(deterministic_values, samples=bootstrap_samples),
            "deterministic_macro_case_rate": statistics.fmean(macro_values),
            "deterministic_macro_case_ci95": bootstrap_mean_ci(macro_values, samples=bootstrap_samples),
            "deterministic_classified": deterministic_classified,
            "exact_secret_leaks": exact_secret_leaks,
            "exact_secret_leak_rate_planned": exact_secret_leaks / len(rows),
            "exact_secret_leak_rate_classified": exact_secret_leaks / deterministic_classified if deterministic_classified else None,
            "judged": judged,
            "judge_id": next(iter(group_judge_ids)) if len(group_judge_ids) == 1 else None,
            "judge_coverage": judged / len(rows),
            "judge_mean_scores": {
                dimension: statistics.fmean(values) if values else None
                for dimension, values in judge_scores.items()
            },
            "role_break_rate": role_breaks / judged if judged else None,
            "secret_leak_rate": secret_leaks / judged if judged else None,
            "ttft_p50_seconds": percentile(ttft, 0.5),
            "ttft_p95_seconds": percentile(ttft, 0.95),
            "total_p50_seconds": percentile(total, 0.5),
            "total_p95_seconds": percentile(total, 0.95),
            "visible_tps_p50": percentile(tps, 0.5),
        }
        summary.append(item)
    return summary


def write_report(
    run_dir: str | Path, *, bootstrap_samples: int = 2000, judge_id: str | None = None
) -> list[dict[str, Any]]:
    directory = Path(run_dir)
    rows = aggregate(directory, bootstrap_samples=bootstrap_samples, judge_id=judge_id)
    write_json_atomic(directory / "summary.json", rows)
    flat_rows = []
    for row in rows:
        flat = {key: value for key, value in row.items() if not isinstance(value, (dict, list))}
        low, high = row["deterministic_macro_case_ci95"]
        flat.update({"deterministic_macro_ci_low": low, "deterministic_macro_ci_high": high})
        for dimension, value in row["judge_mean_scores"].items():
            flat[f"judge_{dimension}"] = value
        flat_rows.append(flat)
    if flat_rows:
        with (directory / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
            writer.writeheader()
            writer.writerows(flat_rows)
    lines = [
        "# NPCBench report",
        "",
        "Failures remain in the denominator. `visible_tps` includes prompt prefill and is not native decode throughput.",
        "",
        "| Model | Mode | Suite | Coverage | Deterministic macro (95% CI) | Exact leak / planned | Judge leak | TTFT p50 / p95 |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        low, high = row["deterministic_macro_case_ci95"]
        macro = row["deterministic_macro_case_rate"]
        ci = f"{macro:.3f} [{low:.3f}, {high:.3f}]" if low is not None else "n/a"
        ttft = "n/a" if row["ttft_p50_seconds"] is None else f"{row['ttft_p50_seconds']:.3f} / {row['ttft_p95_seconds']:.3f}s"
        judged_leak = "n/a" if row["secret_leak_rate"] is None else f"{row['secret_leak_rate']:.1%}"
        lines.append(f"| {row['model_id']} | {row['mode']} | {row['suite']} | {row['coverage']:.1%} | {ci} | {row['exact_secret_leak_rate_planned']:.1%} | {judged_leak} | {ttft} |")
    (directory / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows
