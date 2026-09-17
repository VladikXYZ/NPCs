from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .audit import audit_conversations
from .calibration import calibrate, load_label_csv
from .cases import dataset_fingerprint, load_cases
from .judge import judge_run
from .legacy import migrate_consistency, migrate_jailbreak, write_jsonl
from .report import write_report
from .registry import audit_registry
from .runner import dry_run, run


ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="npcbench", description="Reproducible benchmark for language-model NPCs")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="Validate a case dataset")
    validate.add_argument("dataset", type=Path)

    execute = commands.add_parser("run", help="Run or preview a benchmark config")
    execute.add_argument("config", type=Path)
    execute.add_argument("--output-dir", type=Path, default=ROOT / "benchmarks")
    execute.add_argument("--resume", type=Path)
    execute.add_argument("--dry-run", action="store_true")
    execute.add_argument("--no-weight-hash", action="store_true", help="Faster, less auditable manifest")

    judge = commands.add_parser("judge", help="Judge successful responses with an OpenAI-compatible endpoint")
    judge.add_argument("run_dir", type=Path)
    judge.add_argument("--dataset", type=Path, required=True)
    judge.add_argument("--model", required=True)
    judge.add_argument("--endpoint", required=True, help="Base URL or full /chat/completions URL")
    judge.add_argument("--api-key-env")
    judge.add_argument("--timeout", type=float, default=60)
    judge.add_argument("--retries", type=int, default=2)
    judge.add_argument("--max-tokens", type=int, default=700)

    report = commands.add_parser("report", help="Aggregate a run without dropping failures")
    report.add_argument("run_dir", type=Path)
    report.add_argument("--bootstrap-samples", type=int, default=2000)
    report.add_argument("--judge-id", help="Required when a run contains more than one successful judge")

    audit = commands.add_parser("audit-data", help="Audit a legacy conversation dataset")
    audit.add_argument("dataset", type=Path)
    audit.add_argument("--output", type=Path)

    model_audit = commands.add_parser("audit-models", help="Check model registry release metadata and installed weights")
    model_audit.add_argument("--registry", type=Path, default=ROOT / "models/models.json")
    model_audit.add_argument("--output", type=Path)
    model_audit.add_argument("--strict", action="store_true", help="Exit 1 when release metadata is incomplete")

    migration = commands.add_parser("migrate", help="Convert legacy test definitions to case JSONL")
    migration.add_argument("kind", choices=("consistency", "jailbreak"))
    migration.add_argument("source", type=Path)
    migration.add_argument("output", type=Path)
    migration.add_argument("--npcs", type=Path, default=ROOT / "data_3npcs.json")

    calibration = commands.add_parser("calibrate", help="Compare ordinal human and judge labels")
    calibration.add_argument("labels", type=Path)
    calibration.add_argument("--human-column", default="human_score")
    calibration.add_argument("--judge-column", default="judge_score")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    exit_code = 0
    try:
        if args.command == "validate":
            cases = load_cases(args.dataset)
            payload = {"valid": True, "cases": len(cases), "fingerprint": dataset_fingerprint(cases), "suites": sorted({case["suite"] for case in cases})}
        elif args.command == "run":
            if args.dry_run:
                payload = dry_run(args.config, hash_models=not args.no_weight_hash)
            else:
                directory, counts = run(
                    args.config, args.output_dir, resume=args.resume,
                    hash_models=not args.no_weight_hash,
                )
                payload = {"run_dir": str(directory), "counts": counts}
                if any(count for status, count in counts.items() if status != "ok"):
                    exit_code = 1
        elif args.command == "judge":
            if args.retries < 0 or args.timeout <= 0 or args.max_tokens <= 0:
                raise ValueError("Judge retries must be non-negative; timeout and max-tokens must be positive.")
            payload = judge_run(
                args.run_dir, args.dataset, judge_model=args.model, endpoint=args.endpoint,
                api_key_env=args.api_key_env, timeout_seconds=args.timeout,
                retries=args.retries, max_tokens=args.max_tokens,
            )
        elif args.command == "report":
            if args.bootstrap_samples <= 0:
                raise ValueError("bootstrap-samples must be positive.")
            rows = write_report(args.run_dir, bootstrap_samples=args.bootstrap_samples, judge_id=args.judge_id)
            payload = {"groups": len(rows), "report": str((args.run_dir / "report.md").resolve())}
        elif args.command == "audit-data":
            payload = audit_conversations(args.dataset)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        elif args.command == "audit-models":
            payload = audit_registry(args.registry, ROOT)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            if args.strict and not payload["release_ready"]:
                exit_code = 1
        elif args.command == "migrate":
            cases = (
                migrate_consistency(args.source, args.npcs)
                if args.kind == "consistency"
                else migrate_jailbreak(args.source, args.npcs)
            )
            write_jsonl(args.output, cases)
            payload = {"cases": len(cases), "output": str(args.output.resolve())}
        elif args.command == "calibrate":
            payload = calibrate(load_label_csv(args.labels), args.human_column, args.judge_column)
        else:
            raise AssertionError(args.command)
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return exit_code
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
