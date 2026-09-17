from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Iterable


def weighted_kappa(pairs: Iterable[tuple[int, int]], minimum: int = 1, maximum: int = 5) -> float | None:
    values = list(pairs)
    if not values:
        return None
    levels = list(range(minimum, maximum + 1))
    if any(a not in levels or b not in levels for a, b in values):
        raise ValueError(f"Ratings must be integers in {minimum}..{maximum}")
    observed = Counter(values)
    left = Counter(a for a, _ in values)
    right = Counter(b for _, b in values)
    n = len(values)
    denominator = float((maximum - minimum) ** 2 or 1)
    observed_disagreement = sum(((a - b) ** 2 / denominator) * count / n for (a, b), count in observed.items())
    expected_disagreement = sum(
        ((a - b) ** 2 / denominator) * (left[a] / n) * (right[b] / n)
        for a in levels for b in levels
    )
    if expected_disagreement == 0:
        return 1.0 if observed_disagreement == 0 else None
    return 1 - observed_disagreement / expected_disagreement


def binary_metrics(pairs: Iterable[tuple[bool, bool]]) -> dict[str, float | int | None]:
    values = list(pairs)
    tp = sum(truth and predicted for truth, predicted in values)
    tn = sum(not truth and not predicted for truth, predicted in values)
    fp = sum(not truth and predicted for truth, predicted in values)
    fn = sum(truth and not predicted for truth, predicted in values)
    return {
        "n": len(values), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": (tp + tn) / len(values) if values else None,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
    }


def load_label_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def calibrate(rows: list[dict[str, str]], human_column: str, judge_column: str) -> dict[str, object]:
    score_pairs = [(int(row[human_column]), int(row[judge_column])) for row in rows]
    return {"n": len(rows), "quadratic_weighted_kappa": weighted_kappa(score_pairs)}

