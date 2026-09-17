# Changes since the `vladik` branch

Baseline: local and remote `vladik` point to commit `b8aa3c4` (`minireasoning still doesnt work`).
The changes described here are currently in the `jakub_1` working tree and are not
yet a commit.

## Short version to send to Vladik

We preserved your Vulkan/device loader, speed-test direction and Jinja work, then
fixed the inference/measurement problems around them and built the missing benchmark
pipeline. Plain ChatML and thinking-capable ChatML are now separate; temperature is
applied at generation; warm-up uses the real serializer; empty chunks are ignored;
stream chunks are no longer called tokens; TTFT/dialogue TTFT and failures have
explicit definitions; Supra is hidden from model evaluation; and chat supports NPC,
model and device selection with reset/exit.

On top of that there is now a versioned `npcbench` package: case schemas, 20 public
development cases, multi-turn execution, persistent isolated model workers with a
hard parent timeout, exact-once resume protection, deterministic secret/format
checks, strict JSON LLM judging with cache/retries, human-calibration helpers,
status-aware reports with bootstrap intervals, legacy dataset converters, audits,
configs, documentation and CI tests.

We also tested the seven GGUFs currently installed. Three registry filenames were
wrong and were corrected. Phi-4 and Phi-4-mini use different token grammars, so they
now have separate serializers. The Gemma adapter incorrectly forced a thought-channel
prefix; removing it changed the Gemma template check from one format failure to 12/12
valid outputs across both Gemmas and both modes.

## Concrete fixes to Vladik's runtime

- Split plain `chatml` from explicit `chatml_reasoning`; LFM remains plain while
  Qwen/Bonsai opt into the closed-think prefix.
- Added a distinct `phi_chatml_sep` family because the installed Phi-4 uses
  `<|im_start|>...<|im_sep|>`, while Phi-4-mini uses `<|role|>...<|end|>`.
- Matched the installed Gemma-4 generation prefix to its embedded GGUF template.
- Corrected installed registry paths for Meta-Llama 3.1, Mistral Nemo Q4_K_S and
  Phi-4 Q4_K_S. The benchmark configs explicitly select only the seven installed,
  prepared families.
- Moved temperature/seed/max-token sampling controls to generation instead of model
  construction. Warm-up now uses the same serializer and resets token history.
- Kept raw output, plan and player-visible dialogue separately. Malformed mini output
  is an explicit `invalid_output`, not silently presented as NPC speech.
- Measured first non-whitespace text and first valid dialogue. Retokenized visible
  output is clearly labelled as end-to-end text throughput, not native decode speed.
- Added per-request statuses for timeout, load error, truncation, empty output and
  parsing failure. `-1` is only an unavailable metric sentinel.
- Hid the Supra probe from model selection while retaining it for GPU discovery.
- Removed the false `params: 122` metadata and added an audit that blocks a release
  until artifact source/revision/parameter/template data are verified.
- Removed dependence on the current working directory and personal result paths in
  the new execution/reporting path.

## New benchmark system

- `python -m npcbench validate|run|judge|report|audit-data|audit-models|migrate|calibrate`.
- Versioned JSON/JSONL schemas with stable case/model/mode/attempt keys.
- Twenty annotated public development cases across four fictional worlds and eight
  suites: identity/style, known lore, unknown lore, conflicting information, state,
  memory, robustness and jailbreak/privacy.
- Real multi-turn execution: every user turn receives a model reply. Secret leakage
  on any earlier turn or inside mini-plan raw text cannot be hidden by a final refusal.
- Persistent per-model subprocesses plus a parent-enforced hard timeout. A hung
  native request is killed and the worker can be restarted.
- Manifests record config, dataset, implementation, environment and model provenance.
  Resume rejects changed inputs or duplicate/unplanned result keys.
- Strict semantic-judge JSON, logged attempts, bounded retries and response/rubric
  caching. Multiple judge identities cannot be mixed accidentally in one report.
- Reports keep failures in the denominator and provide coverage, micro/macro rates,
  exact leak rates, p50/p95 latency and deterministic bootstrap intervals.
- Legacy consistency and jailbreak definitions can be migrated without pretending
  their historical outputs have clean provenance.
- `chars.json` audit: 1,689 conversations, 27,024 messages, four empty messages and
  no exact duplicate conversations. Upstream revision/license still needs pinning.
- Automated tests currently pass: `62 passed` after installed-template fixtures are
  included. CI runs validation and mocked tests without requiring model weights.

## Real installed-model admission evidence

Configuration: three public development cases, `off` and `mini`, seed 42, temperature
0, NVIDIA Vulkan device, 20 GPU layers, one trial per condition. This is a serializer/
functionality check, not a statistically meaningful model ranking.

- Initial matrix: `benchmarks/20260916T120538Z-133ab924` — 38/42 valid outputs,
  no load errors, timeouts or crashes.
- Gemma post-fix matrix: `benchmarks/20260916T121914Z-4d3f0da6` — 12/12 valid outputs.
- Combining the corrected Gemma check with unchanged other-family observations gives
  39/42 structurally valid conditions: all 21/21 `off` outputs; mini failures were
  Llama 1/3 and Phi-mini 2/3. Mistral Nemo, Phi-4, Qwen3.5 and both corrected Gemmas
  were clean in both modes for these cases.
- The Llama failure omitted `<speech>`. Phi-mini emitted dialogue before a trailing
  `<speech>`. These are model protocol-compliance failures, not serializer crashes.

## Still to do

1. Obtain and record canonical quantized-artifact source URLs and exact revisions;
   verify incomplete/embedded license metadata, especially Phi-mini. Full GGUF
   SHA-256 hashes, embedded architecture, quantization and chat-template hashes are
   now recorded for the installed seven; the admission manifests themselves skipped
   rehashing during execution and retain `sha256: null` plus registry provenance.
2. Decide whether mini mode is an experimental scored condition with formatting
   failures (recommended) or whether each family gets separately tuned prompts.
   Do not make the parser permissive after seeing failures.
3. Run the full 20-case development matrix, inspect raw failures and calibrate the
   deterministic concept checks. The three-case admission results are not quality scores.
4. Have two people review release cases, add the planned independent characters and
   freeze a world/persona/attack-family held-out split.
5. Label roughly 100–200 responses independently, adjudicate disagreements and
   calibrate the semantic judge. No judge-based leaderboard should be published yet.
6. Add native backend prefill/decode counters, memory telemetry and controlled
   tokenized context-length sweeps. Current visible-token throughput includes prefill.
7. Complete crash/OOM/context-overflow tests with real weights and decide hardware
   settings per model. The 20-layer admission setting was chosen for functionality,
   not peak-speed comparison.
8. Pin the provenance/license of `chars.json`, then either clean and formally split
   it or keep it excluded from scoring.
