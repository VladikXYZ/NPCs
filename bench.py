"""NPC performance smoke runner. See docs/BENCHMARK_PLAN.md for release gates."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import uuid

from npc_runtime import Generation, generate, npc_messages

ROOT = Path(__file__).resolve().parent


def load_prompts(path):
    items = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(items, list) or not items:
        raise ValueError("Prompt dataset must be a nonempty list.")
    prompts = []
    for item in items:
        if isinstance(item, str):
            item = {"id": hashlib.sha256(item.encode()).hexdigest()[:16],
                    "category": "unspecified", "text": item}
        if not isinstance(item, dict) or not all(isinstance(item.get(key), str) and item[key].strip()
                   for key in ("id", "category", "text")):
            raise ValueError("Each prompt needs nonempty id, category and text strings.")
        prompts.append(item)
    if len({item["id"] for item in prompts}) != len(prompts):
        raise ValueError("Prompt IDs must be unique.")
    return prompts


def choose(items, label, describe):
    for i, item in enumerate(items):
        print(f"[{i}] {describe(item)}")
    index = int(input(f"{label} index: "))
    if not 0 <= index < len(items):
        raise ValueError(f"Invalid {label.lower()} index.")
    return items[index]


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("device", nargs="?", type=int, help="Device index; -1 runs each device in its own process.")
    p.add_argument("--reasoning", choices=("off", "mini"), default="off")
    p.add_argument("--npc", default="Garrick")
    p.add_argument("--model", action="append", default=[], help="Model-name substring; repeat to select several.")
    p.add_argument("--prompts", type=Path, default=ROOT / "vlad/performance_prompts.json")
    p.add_argument("--output-dir", type=Path, default=ROOT / "benchmarks")
    p.add_argument("--context-size", type=int, default=4096)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--timeout", type=float, default=60.0, help="Cooperative seconds per request, not a hard watchdog.")
    p.add_argument("--conversation", action="store_true", help="Retain generated replies between prompts.")
    p.add_argument("--dry-run", action="store_true", help="Validate inputs and list installed models without loading weights.")
    return p


def validate_args(args):
    if min(args.context_size, args.max_tokens, args.repeats, args.timeout) <= 0:
        raise ValueError("Context, token limit, repeats and timeout must be positive.")
    if not math.isfinite(args.temperature) or args.temperature < 0:
        raise ValueError("Temperature must be finite and nonnegative.")
    if not math.isfinite(args.timeout):
        raise ValueError("Timeout must be finite.")
    if args.max_tokens >= args.context_size:
        raise ValueError("max-tokens must leave room for the prompt in context-size.")


def sha256(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def run_benchmark(args, device, models, npc, prompts):
    import utils

    os.environ["GGML_VK_VISIBLE_DEVICES"] = str(device["id"]) if device["type"] == "Vulkan" else ""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_dir = args.output_dir / f"{stamp}-{uuid.uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    system = npc_messages(npc, args.reasoning)
    version = importlib.metadata.version("llama-cpp-python")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip()
    manifest = {
        "schema_version": "0.2", "run_id": run_dir.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit, "host": platform.node(), "platform": platform.platform(),
        "python": sys.version, "llama_cpp_python": version, "device": device,
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "models": models, "system_messages": system, "prompts": prompts,
        "source_sha256": {name: sha256(ROOT / name) for name in
                          ("bench.py", "npc_runtime.py", "utils.py", "chat_templates.py", "models/models.json")},
        "metrics": {
            "ttft": "seconds until first non-whitespace streamed text, including plan if present",
            "dialogue_ttft": "seconds until first non-whitespace dialogue after <speech> in mini mode",
            "text_tokens_per_second": "retokenized raw output text / total request seconds, includes prefill; NOT decode throughput",
            "sentinel": "-1 means unavailable, never include it in averages",
            "timeout": "cooperative only; cannot interrupt a stalled native call",
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    failures = 0
    with (run_dir / "responses.jsonl").open("w", encoding="utf-8") as raw_file, \
            (run_dir / "results.csv").open("w", encoding="utf-8", newline="") as csv_file:
        writer = None
        for model in models:
            llm = None
            load_error = ""
            try:
                llm = utils.load_llm(
                    model, {"model_path": model["path"],
                            "n_gpu_layers": -1 if device["type"] == "Vulkan" else 0,
                            "n_ctx": args.context_size, "verbose": False},
                    warmup_inputs=system + [{"role": "user", "content": "Hello."}],
                    custom_jinja=True, reasoning=args.reasoning, log=True,
                )
            except Exception as exc:
                load_error = str(exc)
            try:
                for repeat in range(args.repeats):
                    history = list(system)
                    blocked = load_error
                    if llm:
                        llm.reset()
                    for prompt in prompts:
                        if blocked:
                            result = Generation(status="load_error" if load_error else "skipped", error=blocked)
                        else:
                            if not args.conversation:
                                history = list(system)
                                llm.reset()
                            history.append({"role": "user", "content": prompt["text"]})
                            result = generate(llm, history, reasoning=args.reasoning,
                                              max_tokens=args.max_tokens, temperature=args.temperature,
                                              seed=args.seed, timeout=args.timeout)
                            if args.conversation:
                                if result.status == "ok":
                                    # Preserve raw generated history, including plans; declared in manifest.
                                    history.append({"role": "assistant", "content": result.raw})
                                else:
                                    blocked = f"Previous conversation turn failed: {result.status}"
                        record = {
                            "run_id": run_dir.name, "model": model["name"],
                            "family": model.get("family"), "npc": npc["name"],
                            "reasoning": args.reasoning, "repeat": repeat,
                            "seed": args.seed, "prompt_id": prompt["id"],
                            "category": prompt["category"], "prompt": prompt["text"],
                            **result.as_dict(),
                        }
                        if writer is None:
                            writer = csv.DictWriter(csv_file, fieldnames=list(record))
                            writer.writeheader()
                        writer.writerow(record)
                        raw_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                        csv_file.flush()
                        raw_file.flush()
                        failures += result.status != "ok"
                        print(f"{model['name']} | {prompt['id']} | {result.status} | TTFT {result.ttft:.3f}s")
            finally:
                if llm is not None:
                    llm.close()
    print(f"Saved {run_dir}; {failures} non-ok rows.")
    return 1 if failures else 0


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        validate_args(args)
        prompts = load_prompts(args.prompts)
        npcs = json.loads((ROOT / "data_3npcs.json").read_text(encoding="utf-8"))
        npc = next((npc for npc in npcs if npc["name"].casefold() == args.npc.casefold()), None)
        if npc is None:
            raise ValueError(f"Unknown NPC {args.npc}; choose from {', '.join(n['name'] for n in npcs)}")
        import utils
        models = utils.get_models()
        if args.model:
            for query in args.model:
                if not any(query.casefold() in model["name"].casefold() for model in models):
                    raise ValueError(f"No installed visible model matches {query!r}.")
            models = [model for model in models if any(query.casefold() in model["name"].casefold() for query in args.model)]
        if args.dry_run:
            print(json.dumps({"models": models, "npc": npc["name"], "reasoning": args.reasoning,
                              "prompts": len(prompts), "categories": sorted({p["category"] for p in prompts}),
                              "planned_requests": len(models) * len(prompts) * args.repeats}, indent=2))
            if not models:
                print("No visible NPC model weights installed; the hidden Supra probe is excluded.")
            return 0
        if not models:
            raise ValueError("No visible NPC model weights installed. Add a registered GGUF to models/.")
        devices = utils.get_devices()
        if args.device == -1:
            # Device visibility needs a fresh backend process for each accelerator.
            remaining = list(sys.argv[1:] if argv is None else argv)
            remaining.remove("-1")
            codes = [subprocess.run([sys.executable, str(ROOT / "bench.py"), str(i), *remaining]).returncode
                     for i in range(len(devices))]
            return int(any(codes))
        if args.device is None:
            device = choose(devices, "Device", lambda d: f"{d['type']} | {d['name']}")
        elif 0 <= args.device < len(devices):
            device = devices[args.device]
        else:
            raise ValueError(f"Invalid device index {args.device}.")
        return run_benchmark(args, device, models, npc, prompts)
    except (ValueError, OSError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
