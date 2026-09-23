# NPC Benchmark

NPCBench evaluates local language models as game NPCs across role fidelity, lore,
uncertainty, state tracking, memory, adversarial robustness, privacy and latency.
The repository now has a versioned case contract, one resumable execution path,
hard process isolation, deterministic checks, structured semantic judging,
calibration helpers and status-aware reports. It is a **development benchmark**, not
yet a leaderboard: model artifacts and the held-out human-reviewed set are not frozen.

## Quick start

Use Python 3.12+ and install the pinned runtime dependencies after configuring your
chosen backend below. Large model weights are gitignored and not distributed with
the repository. Place your GGUFs at the paths registered in `models/models.json`.

```powershell
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python -m npcbench validate datasets/dev/cases.jsonl
python -m npcbench run configs/smoke.json --dry-run --no-weight-hash
python -m pytest -q

# After installing registered GGUF weights:
python -m npcbench run configs/smoke.json
python -m npcbench report benchmarks/<run-id>

# Optional structured semantic judge (OpenAI-compatible endpoint):
python -m npcbench judge benchmarks/<run-id> --dataset datasets/dev/cases.jsonl `
  --model <judge-model> --endpoint http://localhost:8000/v1

# Registry and legacy-data gates:
python -m npcbench audit-models
python -m npcbench audit-data chars.json --output docs/chars_audit.json
```

Every configured request gets one durable outcome in `results.jsonl`; timeouts,
format failures and load errors remain in report denominators. A persistent model
worker avoids reloading weights between cases, while the parent process can kill and
restart it after a hard per-case timeout. Resume is accepted only when config,
dataset, implementation and model-artifact fingerprints still match.

The 20 public cases in `datasets/dev/cases.jsonl` are an engineering fixture across
four fictional worlds. They are visible development data and must not be reported as
a held-out benchmark. See [methodology](docs/METHODOLOGY.md), [dataset provenance](datasets/SOURCES.md),
and the [remaining release plan](docs/BENCHMARK_PLAN.md).

The current local admission config selects exactly seven installed/prepared artifacts:
Gemma-4 E2B/E4B, Llama 3.1 8B, Mistral Nemo 12B, Phi-4-mini, Phi-4 and Qwen3.5 9B.
Use `configs/installed-admission.json` for the three-case serializer gate. The real
admission evidence and remaining caveats are summarized in
[the Vladik branch changelog](docs/VLADIK_CHANGES.md).
The checked-in local configs target device index 1 (the NVIDIA Vulkan device on the
current host) with 20 GPU layers so every installed artifact fits; change those
hardware fields explicitly before running on another machine or doing speed claims.

## Compatibility tools

The original performance and interactive tools remain available:

```powershell
python bench.py --dry-run
python chat.py
python chat.py --npc Torin --model "Qwen3.5 4B" --reasoning mini
python bench.py 0 --model "Qwen3.5 4B" --npc Garrick --reasoning off --repeats 5
```

Device indices come from the interactive device list. Omit the positional device
to select interactively; `-1` runs each detected device in a separate process.
Repeat `--model` to select multiple model-name substrings. The hidden Supra router
is used only for device detection; an installation with only Supra has no eligible
NPC models. `--dry-run` performs validation without loading weights or scanning GPUs.

Debug chat chooses a model, device and NPC. `/reset` clears the conversation;
`/exit` quits. The console explicitly shows raw generation for debugging, including
the plan in mini mode, then separately displays the parsed NPC dialogue.

`--context-size`, `--max-tokens`, `--temperature`, `--seed`, `--repeats`, `--timeout`,
`--prompts` and `--output-dir` control experiments. Defaults are warm weights,
isolated prompts with cleared token history, temperature 0, seed 42 and one repeat.
`--conversation` retains prior raw replies within a repeat, then clears history for
the next repeat. Failed conversation turns cause later turns to be marked skipped.
Timing repeats use the same seed; they are not independent behavioral trials.
Increasing context allocation does not itself increase prompt length.

## Reasoning and templates

- `off`: request spoken NPC dialogue only.
- `mini`: request a brief response plan, exactly one `<speech>` separator, and NPC
  dialogue. Save both parts and the raw output. Missing/duplicate separators,
  leaked think tags and empty dialogue are formatting failures.
- `chatml` and the compatibility alias `chatml_nr` serialize plain ChatML.
- `chatml_reasoning` adds a closed-think generation prefix, explicitly selected
  for Qwen/Bonsai based on the local extracted templates.
- `native` delegates serialization to the chat template embedded in the GGUF. The
  admitted LFM2.5 230M uses this because its template includes the model's BOS
  and other model-specific formatting. HF `{% generation %}` annotations are
  removed as inference-only loss-mask markers; their enclosed text is retained.

Mini mode is a prompt intervention, not a universal native-thinking switch. The
requested two-sentence plan length is not a separately enforced token budget.
Other family adapters remain experimental until validated against exact GGUFs;
models with `family: null` use the backend template/default reasoning behavior
but are excluded from benchmark model selection. `family: native` is the
explicit, benchmark-supported opt-in to a GGUF's embedded template.
For those models, `off` is an output instruction, not proof that native thinking
has been disabled. Both modes use the same serializer and warm-up procedure.

## Results and metric definitions

The compatibility `bench.py` creates `manifest.json`, `responses.jsonl` and
`results.csv`. The unified `npcbench` runner creates `manifest.json`, `results.jsonl`
and `run-summary.json`; `npcbench report` derives `summary.json`, `summary.csv` and
`report.md`. The unified manifest records SHA-256 model hashes by default, source
implementation fingerprint, config/dataset fingerprints, environment, device and
all planned exact-once keys.

The compatibility runner's schema `0.2` uses named fields and prompt IDs/categories. The default
`vlad/performance_prompts.json` retains the ten original prompts with explicit
labels. Legacy string-list datasets can be used, but their category is recorded as
`unspecified`; no short/long labels are guessed from row order.

| Field | Meaning |
| --- | --- |
| `ttft` | Request start to first non-whitespace streamed text; includes planning text in mini mode |
| `dialogue_ttft` | Request start to first spoken text after `<speech>` in mini mode |
| `total_time` | Total streamed request duration, excluding later retokenization |
| `output_text_tokens` | Retokenized raw output; excludes hidden and stop tokens |
| `text_tokens_per_second` | Output text tokens divided by total request duration; includes prefill and is **not native decode throughput** |
| `content_chunks` | Diagnostic stream-chunk count, never a token count |
| `status` | `ok`, `empty_response`, `format_error`, `truncated`, `timeout`, `error`, `load_error`, or `skipped` |

`-1` means unavailable, and must be excluded from numeric averages. A failure has
an explicit status and may retain partial text/timing. Report failure counts and
coverage along with successful-response statistics. A run exits 1 if any row is
non-ok; setup/configuration failures exit 2. Dry-run can exit 0 with zero installed
models; check its printed request count before scheduling a real experiment.

Inside a worker, streaming timeouts are cooperative. The unified runner additionally
uses a parent watchdog and terminates the worker if native generation does not return.
Model startup has a separate hard deadline. `bench.py` remains compatibility-only and
still has cooperative request timeouts.

Historical pilot CSVs and presentation charts use a different measurement protocol.
Do not feed schema 0.2 results to the row-position-based scripts in `viz/` or `vlad/`,
or compare the new text-throughput field directly with old `T/s`. Report migration
and native prefill/decode timing are explicit tasks in the plan.

## Verification

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
python bench.py --dry-run --reasoning mini
```

Tests render real Jinja templates and use fake streams for timing, strict parsing,
multi-turn execution, failure denominators and statistics. Seven local GGUFs passed
the load/dialogue gate; combined post-fix admission evidence has 39/42 structurally
valid conditions. That three-case check is not a model ranking, and canonical model
quantized-artifact sources/revisions and complete verified licenses remain release blockers. Full local GGUF and embedded
template hashes are recorded. Historical scripts are
preserved as pilot artifacts and are not used by the unified report path.

## Cross-platform backend setup

This repository contains the configuration and logic to run **llama-cpp-python** with hardware acceleration on all Vulkan supported devices.

## 🚀 Quick Start

### Prerequisites

*   **Python 3.12+**
    *   Ensure Python is installed and added to your system PATH.
*   **Vulkan SDK** (Tested on 1.4.341.1)
    *   [Download here](https://vulkan.lunarg.com/sdk/home) if not installed. Verify your installation with:
    ```bash
    vulkaninfo --summary
    ```
    * Vulkan SDK required components:
      * The Vulkan SDK Core
      * GLM Headers.
      * SDL libraries and headers.
      * Volk header, source, and library.
      * Vulkan Memory Allocator header.
*   **CMake** (Tested on 4.3.2)
    *   [Download here](https://cmake.org/download/) if missing. Verify with:
    ```bash
    cmake --version
    ```
* **venv:** Create virtual environment

### Windows 11
Install llama-cpp-python with this command:
```powershell
$env:CMAKE_ARGS="-DGGML_VULKAN=on"; python -m pip install llama-cpp-python==0.3.21 --force-reinstall --no-cache-dir --no-binary llama-cpp-python
```
<!--
Otherwise add location of VulkanSDK:
```powershell
$env:CMAKE_ARGS="-DGGML_VULKAN=on -DVulkan_SDK='C:\VulkanSDK\1.4.341.1' -DVulkan_INCLUDE_DIR='C:\VulkanSDK\1.4.341.1\Include' -DVulkan_LIBRARY='C:\VulkanSDK\1.4.341.1\Lib\vulkan-1.lib'"; pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir --no-binary llama-cpp-python
```
-->
---

### 🐧 Linux Setup (Ubuntu 24.04 and CachyOS)
Install llama-cpp-python with this command:
```bash
export CMAKE_ARGS="-DGGML_VULKAN=on"
python -m pip install llama-cpp-python==0.3.21 --force-reinstall --no-cache-dir --no-binary llama-cpp-python
```
Install llama-cpp-python using ninja with this command:
```bash
CMAKE_ARGS="-DGGML_VULKAN=on -GNinja" python -m pip install llama-cpp-python==0.3.21 --force-reinstall --no-cache-dir --no-binary llama-cpp-python
```

For a separate CUDA build experiment (the current device selector enumerates Vulkan, not CUDA):
```bash
CMAKE_ARGS="-DGGML_CUDA=on -GNinja" python -m pip install llama-cpp-python==0.3.21 --force-reinstall --no-cache-dir --no-binary llama-cpp-python
```

---
