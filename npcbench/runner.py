from __future__ import annotations

import json
import math
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from npc_runtime import Generation, generate

from .cases import dataset_fingerprint, load_cases, select_cases
from .prompts import case_messages, case_system_message
from .provenance import environment_provenance, implementation_fingerprint, model_id, model_provenance
from .schemas import SCHEMA_VERSION, SUITES, RunKey, fingerprint
from .sandbox import HardRequestTimeout, IsolatedModelSession
from .storage import append_jsonl, assert_unique_results, completed_keys, read_jsonl, write_json_atomic
from .validators import evaluate_deterministic


ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    source = Path(path).resolve()
    config = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Benchmark config must be a JSON object.")
    validate_config(config)
    return config, source


def validate_config(config: dict[str, Any]) -> None:
    required = ("name", "dataset", "modes", "repetitions", "generation")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Config missing fields: {', '.join(missing)}")
    if not isinstance(config["name"], str) or not config["name"].strip():
        raise ValueError("Config name must be non-empty.")
    if not isinstance(config["dataset"], str) or not config["dataset"].strip():
        raise ValueError("Config dataset must be a non-empty path string.")
    modes = config["modes"]
    if (
        not isinstance(modes, list) or not modes
        or not all(isinstance(mode, str) for mode in modes)
        or not set(modes) <= {"off", "mini"}
    ):
        raise ValueError("Config modes must be a non-empty subset of ['off', 'mini'].")
    if len(set(modes)) != len(modes):
        raise ValueError("Config modes must not contain duplicates.")
    for key in ("models", "suites", "case_ids"):
        if key in config and (
            not isinstance(config[key], list)
            or not all(isinstance(value, str) and value.strip() for value in config[key])
        ):
            raise ValueError(f"Config {key} must be a list of non-empty strings.")
    if config.get("suites") and not set(config["suites"]) <= SUITES:
        raise ValueError("Config suites contains an unsupported suite.")
    if not isinstance(config["repetitions"], int) or isinstance(config["repetitions"], bool) or config["repetitions"] <= 0:
        raise ValueError("Config repetitions must be a positive integer.")
    if not isinstance(config.get("device_index", 0), int) or isinstance(config.get("device_index", 0), bool) or config.get("device_index", 0) < 0:
        raise ValueError("device_index must be a non-negative integer.")
    if not isinstance(config.get("isolate_process", True), bool):
        raise ValueError("isolate_process must be boolean.")
    generation = config["generation"]
    if not isinstance(generation, dict):
        raise ValueError("Config generation must be an object.")
    for key in ("worker_start_timeout_seconds", "watchdog_grace_seconds"):
        if key in config and (
            isinstance(config[key], bool)
            or not isinstance(config[key], (int, float))
            or not math.isfinite(config[key]) or config[key] <= 0
        ):
            raise ValueError(f"{key} must be positive.")
    for key in ("context_size", "max_tokens"):
        if not isinstance(generation.get(key), int) or isinstance(generation[key], bool) or generation[key] <= 0:
            raise ValueError(f"generation.{key} must be a positive integer.")
    timeout = generation.get("timeout_seconds")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("generation.timeout_seconds must be finite and positive.")
    if generation["max_tokens"] >= generation["context_size"]:
        raise ValueError("max_tokens must be smaller than context_size.")
    temperature = generation.get("temperature", 0.0)
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature < 0:
        raise ValueError("generation.temperature must be finite and non-negative.")
    if not isinstance(generation.get("seed", 42), int) or isinstance(generation.get("seed", 42), bool):
        raise ValueError("generation.seed must be an integer.")
    gpu_layers = generation.get("gpu_layers", -1)
    if not isinstance(gpu_layers, int) or isinstance(gpu_layers, bool) or gpu_layers < -1:
        raise ValueError("generation.gpu_layers must be -1 (all) or a non-negative integer.")


def _resolve_project_path(value: str, config_path: Path) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    beside = (config_path.parent / candidate).resolve()
    return beside if beside.exists() else (ROOT / candidate).resolve()


def select_models(installed: list[dict[str, Any]], selectors: list[str] | None) -> list[dict[str, Any]]:
    from chat_templates import INFERENCE_TYPES, NATIVE_TEMPLATE_FAMILY

    supported_families = set(INFERENCE_TYPES[0]) | {NATIVE_TEMPLATE_FAMILY}
    supported = [model for model in installed if model.get("family") in supported_families]
    if not selectors:
        return supported
    selected: list[dict[str, Any]] = []
    for selector in selectors:
        matches = [
            model for model in supported
            if selector.casefold() in model["name"].casefold() or selector == model_id(model)
        ]
        if not matches:
            unsupported = [model["name"] for model in installed if selector.casefold() in model["name"].casefold()]
            if unsupported:
                raise ValueError(f"Installed model matches {selector!r} but has no prepared serializer family: {', '.join(unsupported)}")
            raise ValueError(f"No installed visible model matches {selector!r}.")
        for model in matches:
            if model not in selected:
                selected.append(model)
    return selected


def planned_keys(cases: list[dict[str, Any]], models: list[dict[str, Any]], modes: list[str], repetitions: int) -> list[RunKey]:
    return [
        RunKey(case["case_id"], model_id(model), mode, attempt)
        for model in models
        for mode in modes
        for attempt in range(repetitions)
        for case in cases
    ]


def generate_case(
    llm: Any, case: dict[str, Any], mode: str, generation: dict[str, Any], seed: int
) -> tuple[Any, list[dict[str, Any]], list[dict[str, str]], list[str]]:
    """Run every user turn, preserving model replies as conversation history."""
    history: list[dict[str, str]] = [case_system_message(case, mode)]
    turn_measurements: list[dict[str, Any]] = []
    generated_raw_turns: list[str] = []
    last = None
    for message in case["messages"]:
        history.append(message)
        if message["role"] != "user":
            continue
        last = generate(
            llm,
            history,
            reasoning=mode,
            max_tokens=int(generation["max_tokens"]),
            temperature=float(generation.get("temperature", 0.0)),
            seed=seed,
            timeout=float(generation["timeout_seconds"]),
        )
        turn_measurements.append({
            "status": last.status,
            "ttft_seconds": last.ttft,
            "dialogue_ttft_seconds": last.dialogue_ttft,
            "total_seconds": last.total_time,
            "visible_output_tokens": last.output_text_tokens,
            "text_tokens_per_second": last.text_tokens_per_second,
            "finish_reason": last.finish_reason,
        })
        if last.raw:
            generated_raw_turns.append(last.raw)
            history.append({"role": "assistant", "content": last.raw})
        if last.status != "ok":
            break
    if last is None:
        raise ValueError(f"Case {case['case_id']} has no user turn")
    return last, turn_measurements, history, generated_raw_turns


def prepare_run(
    config_path: str | Path,
    output_root: str | Path,
    *,
    resume: str | Path | None = None,
    hash_models: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], Path, dict[str, Any]]:
    config, source = load_config(config_path)
    cases = load_cases(_resolve_project_path(config["dataset"], source))
    suites = set(config.get("suites", [])) or None
    case_ids = set(config.get("case_ids", [])) or None
    cases = select_cases(cases, suites=suites, case_ids=case_ids)
    if not cases:
        raise ValueError("No cases remain after suite selection.")

    import utils

    models = select_models(utils.get_models(), config.get("models"))
    if not models:
        raise ValueError("No visible NPC model weights are installed; the hidden hardware probe is not benchmark-eligible.")
    model_records = [model_provenance(model, hash_weights=hash_models) for model in models]
    implementation = implementation_fingerprint(ROOT)
    if resume:
        run_dir = Path(resume).resolve()
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        if manifest["config_fingerprint"] != fingerprint(config):
            raise ValueError("Resume refused: config fingerprint differs from manifest.")
        if manifest["dataset_fingerprint"] != dataset_fingerprint(cases):
            raise ValueError("Resume refused: dataset fingerprint differs from manifest.")
        if manifest.get("implementation_fingerprint") != implementation:
            raise ValueError("Resume refused: benchmark implementation differs from manifest.")
        if manifest.get("models") != model_records:
            raise ValueError("Resume refused: installed model artifacts differ from manifest.")
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = Path(output_root).resolve() / f"{stamp}-{uuid.uuid4().hex[:8]}"
        run_dir.mkdir(parents=True, exist_ok=False)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_dir.name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config_path": str(source),
            "config": config,
            "config_fingerprint": fingerprint(config),
            "dataset_fingerprint": dataset_fingerprint(cases),
            "case_count": len(cases),
            "environment": environment_provenance(ROOT),
            "models": model_records,
            "implementation_fingerprint": implementation,
            "planned_keys": [key.as_tuple() for key in planned_keys(cases, models, config["modes"], config["repetitions"])],
            "metric_definitions": {
                "ttft_seconds": "Wall time to first non-whitespace streamed text, including prefill and planning.",
                "dialogue_ttft_seconds": "Wall time to first valid dialogue after <speech>; -1 means unavailable.",
                "text_tokens_per_second": "Retokenized visible output divided by total request time; not native decode-only throughput.",
            },
        }
        write_json_atomic(run_dir / "manifest.json", manifest)
    return config, cases, models, run_dir, manifest


def run(
    config_path: str | Path,
    output_root: str | Path,
    *,
    resume: str | Path | None = None,
    retry_failures: bool = False,
    hash_models: bool = True,
    loader: Callable[..., Any] | None = None,
) -> tuple[Path, dict[str, int]]:
    if retry_failures:
        raise ValueError("Automatic target-response retries are disabled: failures are benchmark outcomes. Start a new run or increase declared repetitions.")
    config, cases, models, run_dir, manifest = prepare_run(
        config_path, output_root, resume=resume, hash_models=hash_models
    )
    import utils

    loader = loader or utils.load_llm
    results_path = run_dir / "results.jsonl"
    existing = read_jsonl(results_path, validate=True)
    seen = assert_unique_results(existing)
    expected = {tuple(key) for key in manifest["planned_keys"]}
    unexpected = seen - expected
    if unexpected:
        raise ValueError(f"Results contain keys absent from the manifest: {sorted(unexpected)[:3]}")
    done = completed_keys(existing, retry_failures=retry_failures)
    generation = config["generation"]
    isolate_process = bool(config.get("isolate_process", True))
    worker_start_timeout = float(config.get("worker_start_timeout_seconds", 600))
    watchdog_grace = float(config.get("watchdog_grace_seconds", 5))
    device_index = int(config.get("device_index", 0))
    devices = utils.get_devices()
    if not 0 <= device_index < len(devices):
        raise ValueError(f"device_index {device_index} is outside 0..{len(devices)-1}")
    device = devices[device_index]
    manifest["device"] = device
    manifest["isolate_process"] = isolate_process
    write_json_atomic(run_dir / "manifest.json", manifest)
    os.environ["GGML_VK_VISIBLE_DEVICES"] = str(device["id"]) if device["type"] == "Vulkan" else ""
    counts: dict[str, int] = {"ok": 0, "timeout": 0, "error": 0, "invalid_output": 0, "skipped": 0}

    for model in models:
        mid = model_id(model)
        for mode in config["modes"]:
            pending = [
                (attempt, case)
                for attempt in range(config["repetitions"])
                for case in cases
                if RunKey(case["case_id"], mid, mode, attempt).as_tuple() not in done
            ]
            if not pending:
                continue
            llm = None
            session = None
            load_error = ""
            first_user = next(message for message in pending[0][1]["messages"] if message["role"] == "user")
            warmup = [case_system_message(pending[0][1], mode), first_user]
            load_started = time.perf_counter()
            try:
                if isolate_process:
                    session = IsolatedModelSession(model, device, mode, generation, warmup)
                    session.start(worker_start_timeout)
                else:
                    llm = loader(
                        model,
                        {
                            "model_path": model["path"],
                        "n_gpu_layers": int(generation.get("gpu_layers", -1)) if device["type"] == "Vulkan" else 0,
                            "n_ctx": int(generation["context_size"]),
                            "verbose": False,
                        },
                        warmup_inputs=warmup,
                        custom_jinja=True,
                        reasoning=mode,
                        log=True,
                    )
            except Exception as exc:
                load_error = str(exc)
            load_seconds = time.perf_counter() - load_started
            try:
                for attempt, case in pending:
                    messages = case_messages(case, mode)
                    transcript = messages
                    generated_raw_turns: list[str] = []
                    seed = int(generation.get("seed", 42)) + attempt
                    if isolate_process and session is None and not load_error:
                        try:
                            session = IsolatedModelSession(model, device, mode, generation, warmup)
                            session.start(worker_start_timeout)
                            load_seconds = session.load_seconds
                        except Exception as exc:
                            load_error = str(exc)
                    if load_error:
                        status, error, response, raw, measurements = "error", load_error, "", "", {}
                    else:
                        try:
                            if isolate_process:
                                payload = session.generate(
                                    case, seed, float(generation["timeout_seconds"]) + watchdog_grace
                                )
                                generated = Generation(**payload["generation"])
                                turn_measurements = payload["turns"]
                                transcript = payload["transcript"]
                                generated_raw_turns = payload["generated_raw_turns"]
                            else:
                                llm.reset()
                                generated, turn_measurements, transcript, generated_raw_turns = generate_case(
                                    llm, case, mode, generation, seed
                                )
                        except HardRequestTimeout as exc:
                            status, error, response, raw, measurements = "timeout", str(exc), "", "", {}
                            session = None
                            generated = None
                        except Exception as exc:
                            status, error, response, raw, measurements = "error", str(exc), "", "", {}
                            generated = None
                        if generated is not None:
                            status_map = {
                                "ok": "ok", "timeout": "timeout", "error": "error",
                                "format_error": "invalid_output", "empty_response": "invalid_output",
                                "truncated": "invalid_output",
                            }
                            status = status_map.get(generated.status, "error")
                            error = generated.error or (generated.status if status != "ok" else "")
                            response, raw = generated.dialogue, generated.raw
                            measurements = {
                                "ttft_seconds": generated.ttft,
                                "dialogue_ttft_seconds": generated.dialogue_ttft,
                                "total_seconds": generated.total_time,
                                "visible_output_tokens": generated.output_text_tokens,
                                "text_tokens_per_second": generated.text_tokens_per_second,
                                "finish_reason": generated.finish_reason,
                                "turns": turn_measurements,
                            }
                    checks = evaluate_deterministic(case, response, generated_raw_turns) if not load_error and generated_raw_turns else None
                    record = {
                        "schema_version": SCHEMA_VERSION,
                        "run_id": manifest["run_id"],
                        "case_id": case["case_id"],
                        "suite": case["suite"],
                        "model_id": mid,
                        "mode": mode,
                        "attempt": attempt,
                        "seed": seed,
                        "status": status,
                        "error": error,
                        "response": response,
                        "raw_response": raw,
                        "transcript": transcript,
                        "generated_raw_turns": generated_raw_turns if not load_error else [],
                        "measurements": measurements,
                        "deterministic_checks": checks,
                        "load_seconds": load_seconds,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    }
                    append_jsonl(results_path, record)
                    counts[status] += 1
                    print(f"{mid} | {mode} | {case['case_id']} | {status}", flush=True)
            finally:
                if llm is not None:
                    llm.close()
                if session is not None:
                    session.close()
    all_results = read_jsonl(results_path, validate=True)
    recorded_keys = assert_unique_results(all_results)
    total_counts = {
        status: sum(row["status"] == status for row in all_results)
        for status in counts
    }
    write_json_atomic(run_dir / "run-summary.json", {
        "counts": total_counts,
        "planned_requests": len(expected),
        "recorded_requests": len(recorded_keys),
        "complete": recorded_keys == expected,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    })
    return run_dir, total_counts


def dry_run(config_path: str | Path, *, hash_models: bool = False) -> dict[str, Any]:
    config, source = load_config(config_path)
    cases = select_cases(
        load_cases(_resolve_project_path(config["dataset"], source)),
        suites=set(config.get("suites", [])) or None,
        case_ids=set(config.get("case_ids", [])) or None,
    )
    import utils

    models = select_models(utils.get_models(), config.get("models"))
    return {
        "config": config["name"],
        "dataset": str(_resolve_project_path(config["dataset"], source)),
        "dataset_fingerprint": dataset_fingerprint(cases),
        "cases": len(cases),
        "suites": sorted({case["suite"] for case in cases}),
        "models": [model_provenance(model, hash_weights=hash_models) for model in models],
        "modes": config["modes"],
        "repetitions": config["repetitions"],
        "planned_requests": len(planned_keys(cases, models, config["modes"], config["repetitions"])),
    }
