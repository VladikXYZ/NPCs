# NPC benchmark: research and implementation plan

Review date: 2026-09-16. Starting revision: `b8aa3c4`.

Implementation status: the repository now contains the `npcbench` package, the 0.1
case/judgment schemas, 20 public development cases, legacy adapters, persistent
subprocess workers with hard request timeouts, exact-once resume checks, structured
judge logging/cache, calibration statistics, status-aware aggregation/bootstrap
intervals, dataset/model audits and mocked CI. Seven installed GGUFs now have a
three-case real admission pass and recorded weight/template hashes. The remaining
gates are canonical artifact sources/revisions/licenses, broader real-GGUF testing,
independent case review, held-out data, human calibration and native prefill/decode
instrumentation.

## Objective and scientific contribution

Answer: **Which model, prompt policy and hardware configuration can produce useful,
consistent NPC dialogue within a game's response-time and memory budget?**

The strongest direction is the intersection of character quality, resistance to
player manipulation and consumer-hardware latency. Role-playing benchmarks already
exist; novelty must be demonstrated against them, not assumed from an absence of
related work. Candidate contributions are quality under latency constraints,
authoritative world-state adherence, multi-turn retention and robust in-character
handling of adversarial players. This is a research proposal, not a novelty claim.

Keep a vector of results. A fast repeated sentence, universal refusal or eloquent
hallucination must not look like a successful NPC. Eventually report the fraction
of responses that meet both predeclared quality criteria and a dialogue-latency
budget, accompanied by the underlying dimensions and failure counts.

## Current repository findings

| Area | Evidence in this checkout | Consequence / next action |
| --- | --- | --- |
| Runtime | `bench.py`, `utils.py`, `chat_templates.py`; recent commits added mini reasoning | Preserve Vladik's Vulkan/device work; validate adapters against exact model artifacts before rankings. |
| Sampling | Temperature was passed to `Llama(...)`, while generation used its default | The intended deterministic setting was not applied. Now passed to generation explicitly; old runs retain historical status. |
| Serialization | Legacy ChatML forced closed `<think>` tags; plain `chatml_nr` existed but only a few models used it | Plain ChatML is now the default; Qwen/Bonsai have an explicit opt-in family. |
| Measurement | Stream chunks counted as tokens, TTFT could start on empty content, timeout covered the whole model run | New smoke runner handles these cases and clearly labels text-token throughput; native token timing still needs instrumentation. |
| Analysis | `vlad/tables.py` and `viz/testing.py` infer model identity/category from row position; `vlad/test.json` now starts with long prompts | New runs have explicit IDs/categories and a new schema. Historical graphs must not ingest them without migration. |
| Model registry | The false `params: 122` placeholders were removed; quantization, source revision and architecture are still unverified | `npcbench audit-models --strict` now blocks release until verified total/active parameters and artifact provenance are supplied. |
| Quality generation | `jakub/dataGeneration.py` produces JSON, while grading scripts expect CSV transcripts in personal directories | A reproducible generation-to-judge path must be restored; the current quality outputs are historical artifacts. |
| Jailbreak execution | Attack JSON and evaluators exist; no active tracked producer connects the full suite to the new runner | Implement a suite adapter using the shared inference layer. `martin/m.py` is a scratch file, not an entry point. |
| Pilot sample size | 168 consistency grades = 7 models x 3 NPCs x 4 categories x 2 prompt modes | A category/model/mode heatmap cell has only 3 observations. Treat fine rank differences as exploratory. |
| Judges | One model, loosely parsed text, short completion budgets, no adjudicated calibration set | Validate structured outputs, log failures, and calibrate against independent human labels. |
| Attack denominator | Historical log has 735 attempts, 265 successes, 453 secure verdicts, 17 unknown/errors; exact-line analysis recognizes 716 verdicts | Reconcile parsing, disclose coverage, rerun only missing judgments; never count unknowns as secure. |
| Character corpus | `chars.json`: 1,689 conversations, each 16 messages, with 4 empty contents | Potential source material, not yet a validated benchmark. `trash.py` suggests the upstream dataset, but does not prove a pinned provenance match. |
| Installed weights | Seven candidate GGUFs plus the Supra probe are present locally | All seven load and pass dialogue-only admission; mini formatting is imperfect for Llama/Phi-mini. The three-case gate is not a quality ranking. |

## Changes implemented in this review

- Separate `chatml` / legacy `chatml_nr` from explicit `chatml_reasoning`.
  Registry selection is explicit; it is not based on parameter count.
- Apply `off` and `mini` consistently to debug chat and speed testing; retain the
  existing boolean `reason` compatibility in the handler/loading functions.
- Build one NPC system message including persona, existing character constraints
  and the mode's output policy. The native-template fallback receives it too.
- Define mini mode as a brief generated response plan followed by `<speech>` and
  NPC dialogue. It is a prompted experiment, not evidence of internal reasoning
  or a guaranteed native thinking toggle. Planning length is a prompt request,
  not an enforced separate token budget.
- Warm up using the same inference template and a valid user turn; reset token
  history afterwards. Close failed instances without loading a diagnostic copy.
- Hide Supra from benchmark/chat selection while retaining it for device discovery.
- Add NPC selection to chat, raw-output diagnostics, `/reset`, and `/exit`.
- Add CLI controls, dry-run validation, repeat count and isolated/conversation modes.
- Store named, categorized prompts, raw output, plan, dialogue, finish/status fields,
  per-request text/dialogue latency, manifest metadata, and source hashes. Flush
  each row to a unique run directory. Keep `-1` only as an unavailable metric sentinel.
- Stop treating streamed chunks as tokens. The new text-token throughput includes
  request/prefill time; it cannot be substituted for historical `T/s`.
- Add regression tests with real Jinja rendering and synthetic backend streams.

These changes form a testable foundation. In the subsequent implementation pass,
the unified schema/runner/judge/report pipeline and parent watchdog were also added.
They do not establish compatibility for every model in `models/models.json` and do
not convert public development fixtures into a release benchmark.

## Milestone 1 — Validate the inference layer (P0)

Suggested lead: Vladik. Estimated effort: 2–4 focused engineering days after weights
are available. This milestone blocks scored experiments.

1. Smoke-test LFM2.5 230M first, then Qwen3.5 4B, one Gemma, Phi, Llama and edge model
   already available to the team. Use the exact GGUFs that will enter the benchmark.
2. Record repository/revision, SHA-256, quantization, architecture, total/active
   parameters, tokenizer and chat-template hash for each model. Exclude the routing
   probe and unsupported architectures. Q1 versus Q4 is a configuration comparison,
   not a controlled parameter-size study.
3. Render and inspect system + user and system + user + assistant + user prompts.
   Compare each adapter to that artifact's embedded/native template, including BOS,
   EOS and reasoning control. Generic family names do not prove compatibility.
   Audit `gpt oss` assigned to ChatML and Phi variants assigned to one serializer.
4. Run 3 NPCs x 3 prompts (identity, known fact, unknown fact) x 2 modes. Retain raw
   streams, finish reasons and parsing results; do not hide refusals or empty output.
5. Subprocess isolation and parent watchdogs for loading, warm-up and per-case
   generation are implemented. Unit tests cover parent timeout handling; real-GGUF
   admission must still validate process crash, context overflow and GPU OOM recovery.

Acceptance: every admitted model has a reviewed template fixture and a saved real
smoke run. Every requested case has exactly one outcome, including failures. An
unsupported model/configuration has an explicit reason and never a quality score.

## Milestone 2 — Define the dataset contract and game rules (P0)

Suggested leads: Jakub for lore/personas, Martin for adversarial cases. Estimated
effort: 4–7 days. Work on this can proceed alongside Milestone 1.

Use a versioned case schema containing `case_id`, `world_id`, `npc_id`, `suite`,
`scenario_id`, initial world state, public/private facts, allowed actions, dialogue
history/player turns, target behavior, prohibited behavior and scoring criteria.
Separate authoritative game state, NPC beliefs and player assertions.

Start with the existing three original characters as development fixtures; add
at least 12 newly authored characters across several independent worlds for a
release pilot. Decide the final count using pilot variance and review capacity.

| Suite | What a case should test | Example of an objective check |
| --- | --- | --- |
| Known lore | Accurate answers to facts the NPC may reveal | Required fact present, contradictory fact absent |
| Unknown lore | Appropriate uncertainty and useful response | Does not invent an unsupported quest location |
| Identity/style | Persona expressed where relevant | No assistant identity; human-rated voice consistency |
| Conflicting beliefs | Maintain explicitly coexisting motivations | Acknowledges both specified motivations |
| State update | Respect authoritative changes without accepting player fabrications | Inventory/event answer reflects the current state |
| Memory | Recall earlier facts with distractors and delayed queries | Correct event recalled after 5/10/20 turns |
| Robustness | Resist a defined override while remaining a useful NPC | Secret retained AND benign player need addressed |
| Tools (later) | Respect game action permissions | Simulated action validated against game state |

Correct the current construct definitions: a villain adopting kittens is not by
itself a logical contradiction or evidence of psychological discomfort. A character
can love someone and seek to kill them. Specify whether the task requires tension,
multiple motives or a genuine inconsistent fact; do not punish plausible roleplay
for failing to display an emotion absent from the persona. The instruction to
accept unusual world facts must agree with a collision rubric that currently also
rewards rejecting them.

Define public lore versus secrets explicitly. Reciting an ordinary knowledge-base
fact is not automatically leakage. Add synthetic secret canaries and disclosure
rules, and combine exact matching with semantic review for paraphrased leakage.
Include useful benign controls so refusing everything cannot win the benchmark.

Audit `chars.json`: identify/pin upstream revision and reuse terms, validate roles,
empty messages and duplicates, and separate persona descriptions, player queries
and reference assistant replies. Do not provide future reference answers to the
model. Replaying scripted player turns requires an audit for assumptions about
reference replies; a responding player simulator is a separate experimental track.
The likely source is [NPC-Dialogue_v2](https://huggingface.co/datasets/chimbiwide/NPC-Dialogue_v2),
as referenced in local `trash.py`; exact lineage remains to be verified.

Split by world/persona and attack-template family, not random rows. Keep paraphrases
and near-duplicate personas in one split. Freeze a public development set and a
held-out evaluation set; tune prompts only on development data. Novel fictional
worlds reduce dependence on memorized famous-character knowledge.

Acceptance: schema validation passes; two team members review each release case;
each case has clear expected behavior; split overlap checks and provenance are saved.

## Milestone 3 — Calibrate and harden scoring (P0)

Suggested leads: Jakub/Martin, with all three members labeling calibration examples.
Estimated effort: 4–7 days once the first cases exist.

1. Start with deterministic validators for schema/format, synthetic secrets and
   simulated action permissions. Use an LLM judge for semantic judgments only.
2. Score separate fields: role break, lore contradiction, secret disclosure,
   instruction override, benign helpfulness and dialogue quality. Content policy
   should be explicit per game/rating; do not equate an evil fictional persona with
   general unsafe behavior. Tool security claims require actual sandboxed actions.
3. Grade player-visible dialogue without showing the plan or model identity. Audit
   raw output separately for leakage/format failures. Otherwise mini mode can gain
   credit from its explanation while emitting poor dialogue.
4. Use a versioned JSON schema with verdicts, evidence spans and concise rationales.
   Treat transcripts as untrusted data. Test instructions aimed at the judge, fake
   result tags and convincing but unsupported rationales.
5. Persist every judgment attempt with judge identity/version, rubric hash, token
   budget and parse/error status. Retry transient and formatting errors with limits;
   cache by response + rubric + judge hash. Never rerun a successful target response
   just because grading failed.
6. Two independent humans label a stratified calibration set of approximately 100–200
   responses; adjudicate disagreements. Include failures, terse refusals, long answers
   and both modes. Use a second independent judge on calibration/disagreements;
   expand to all results only if budget and observed disagreement warrant it.
7. Report agreement/confusion matrices per dimension, weighted kappa for ordinal
   grades and false-negative rates for secret/tool failures. Choose release criteria
   after the pilot but before evaluating the held-out set; report uncertainty and
   human disagreement instead of imposing an arbitrary universal agreement threshold.

For attacks, publish classified successes / valid classified attempts together with
total planned attempts, inference failures and judge failures. Compare models on
matched cases and show missingness bounds when coverage differs. Report per-turn
outcomes, any-turn success and turns to first success separately. Match attack
turn/token budgets when comparing attack types. Include both micro-averages and
equal-category macro-averages with their weighting policy declared.

Acceptance: a calibration report exists; all outputs validate or have explicit
failure records; changing whitespace cannot change the score denominator.

## Milestone 4 — Make latency claims experimentally defensible (P1)

Suggested lead: Vladik. Estimated effort: 4–6 days after Milestone 1.

- Distinguish cold load, warm model with empty token history, cached-prefix requests
  and growing conversations. The current isolated mode resets token history but
  leaves weights warm; label it accordingly.
- Instrument actual prompt/generated token IDs and timing from the backend. Measure
  prefill and decode separately; cross-check a subset against backend performance
  counters. Include generated stop/hidden tokens under a documented convention.
- Measure TTFT, first usable NPC dialogue, first completed sentence, total completion
  latency, output length, memory use and runtime failures. For spoken NPCs, add TTS
  first-audio latency as a later independent stage.
- Sweep tokenized context lengths (e.g. 256/1k/4k and supported larger contexts),
  keeping the queried facts and answer task controlled. Increasing `n_ctx` allocation
  alone does not create a longer prompt. Log actual tokenizer-specific lengths.
- First microbenchmark equal output budgets; then evaluate natural stopping on real
  NPC cases with answer quality. Record truncation; never force arbitrary output
  length into a quality evaluation without labeling the condition.
- Use paired prompt/model comparisons; randomize or counterbalance order with a saved
  seed. Repeat timing at least 5–10 times and expand as needed. Ten observations are
  insufficient for a stable per-case p95 claim: publish sample count/intervals and
  acquire enough measurements at the reporting stratum to estimate tails.
- Timing repeats use fixed sampling controls. Behavioral trials use separate seeds
  at a declared nonzero temperature if testing stochastic variability. Repeated
  greedy replies do not increase the number of independent behavior scenarios.
- Record OS/backend/driver, CPU threads, power mode, GPU layers, context/KV settings,
  memory, background load and model hashes. Check the unexpectedly large i9 CPU/iGPU
  difference in the pilot before interpreting it as a general hardware conclusion.
- Report p50/p95 dialogue latency and 95% intervals. Cluster resampling by independent
  scenario/world for quality; do not treat turns or repeated seeds as independent
  characters. Use paired differences and avoid unsupported precise rank ordering.

The existing 300/800 ms and 18/6 token/s targets are declared design choices. Validate
them with a small player study before presenting them as perceptual thresholds.
Token/s varies with tokenizer and language; report visible characters/s or words/s
as an additional experience measure. Keep English and Czech results stratified.

Acceptance: raw observations reproduce the report, token counters agree within the
declared convention, and configuration/coverage/uncertainty accompany every chart.

## Milestone 5 — Unify execution and reporting (P1)

Suggested shared engineering work. Estimated effort: 3–5 days.

Evolve the tested helpers into an `npcbench` package after the schema is stable;
retain `bench.py` and `chat.py` as compatibility entry points. Add suite adapters
for the consistency and attack datasets. One CLI/config should support validation,
generation, judging and reports independently. Add requirements/lock manifests for
the chosen backend builds; identical Python package versions alone do not pin
llama.cpp build flags and GPU drivers.

Use JSONL as the durable record, CSV as an export, and plots as derived artifacts.
Add exact-once resume keys based on case/model/config/version/repeat. Resume must
reject mismatched manifests. Preserve raw partial output and planned cases after
crashes. Add a noninteractive CPU/mocked CI job and optional real-GGUF integration
tests for every admitted serializer. Do not commit credentials or large weights.

Migrate the report readers to explicit model/case IDs and status-aware aggregation.
Retain original pilot CSVs untouched. Do not infer labels for ambiguous historical
blocks; mark unknown provenance and document any verified conversion.

Acceptance: a fresh checkout can reproduce a smoke run and regenerate its summaries
without editing paths; interrupt/resume produces no missing or duplicate cases.

## Milestone 6 — Freeze and release the first benchmark (P2)

1. Freeze dataset/rubric/sampler/template/backend revisions before the full run.
2. Start with a verified subset of the team's intended models. Preserve the seven
   historical models as an optional comparison cohort; do not require models that
   cannot pass the backend/template gate. Scale models after coverage is trustworthy.
3. Run predefined baseline comparisons: correct versus shuffled persona, generic
   assistant, refuse-everything control and repeated-sentence control. These should
   expose evaluators that reward superficial style or safe but useless answers.
4. Compare off versus mini on the same cases and dialogue quality rubric, measuring
   planning overhead, formatting failure rate and first-dialogue latency. Native
   thinking, equal-compute budgets and prompt optimization are separate ablations.
5. Publish score profiles, matched-case comparisons, uncertainty, coverage, memory
   and the quality/latency tradeoff. Label empirical non-dominated configurations as
   tentative when intervals overlap. Provide real successes and failures with IDs.
6. Release the specification, data card, model/runtime manifests, raw transcripts,
   adjudicated calibration subset and report-generation command. Refresh the deck
   after results freeze; distinguish historical pilot findings from new evidence.

Acceptance: another team member reproduces a sampled run and every displayed number
is traceable to case IDs and recorded outputs. Finish all P0 gates before a leaderboard.

## The next concrete team milestone

Within the next working week, aim for **two verified models x three NPCs x three
smoke prompts x two modes = 36 generation cases** on one machine, with raw logs and
manual inspection. Add repeat timing only after these generation cases work. In
parallel, agree the dataset schema and draft 20 fully annotated development cases.
This is a functionality gate, not a statistically meaningful model ranking.

Before scaling, estimate cost from measured runs:

`generation work = cases x models x prompt modes x behavioral seeds`

`judge work = completed outputs x judges x grading repeats`

Hardware/timing repetitions multiply only the appropriate experiment. Run a small
pilot to estimate time, memory and judge cost, then select the full experiment size.
Milestone durations above are effort estimates, not commitments or scheduled jobs.

## Related work to use in the specification

These sources establish useful comparison points, not proof that this proposal is novel.

- [RoleLLM / RoleBench](https://arxiv.org/abs/2310.00746): role-specific knowledge and
  speaking-style evaluation. Compare the scope and design of our controlled fictional worlds.
- [CharacterBench](https://arxiv.org/abs/2412.11912): multidimensional character evaluation
  with targeted prompts and human annotations. Use it to improve case/rubric design.
- [RPEval](https://arxiv.org/abs/2505.13157): emotional understanding, decisions, moral
  alignment and character consistency. Distinguish persona alignment from game policy.
- [RPGBench](https://arxiv.org/abs/2502.00595): models as RPG engines. Position our
  individual-NPC evaluation against broader game-engine evaluation.
- [Judging LLM-as-a-Judge](https://arxiv.org/abs/2306.05685): judge biases and human
  agreement. Motivate blinding, length controls and empirical calibration.
- [LFM2.5-230M native template](https://huggingface.co/LiquidAI/LFM2.5-230M/blob/main/chat_template.jinja):
  primary template reference; pin the artifact revision before admitting a model.

The project's local `extracted_templates.txt` is also evidence for serializer
debugging; it needs per-artifact hashes and extraction dates before release use.
