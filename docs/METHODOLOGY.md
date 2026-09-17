# NPCBench methodology (development version 0.1)

## Unit of evaluation

A case fixes a fictional world, NPC persona, public and private facts, permitted
actions, player turns and target behavior. A trial is identified by
`(case_id, model_id, mode, attempt)`. The runner answers each player turn in order;
only the final player-visible dialogue is scored against the case target, while the
complete transcript and raw final output are retained for audit.

`off` requests dialogue only. `mini` requests a short plan followed by exactly one
`<speech>` separator. The judge sees player-visible dialogue, not the plan or model
identity. Raw output is checked separately for format failures.

## Scoring

Deterministic validators cover only high-precision facts: required literal concept
groups, forbidden phrases, synthetic secret canaries, obvious assistant-identity
breaks and the declared sentence limit. They retain matched evidence. These checks
are not semantic graders.

The optional judge returns strict JSON with six 1–5 dimensions: character fidelity,
world consistency, relevance, style, privacy and action grounding. It also records
role break, secret leak, unsupported claims, response excerpts and a rationale. It
scores quality from the final player-visible dialogue while auditing all raw assistant
turns (including mini-plan text) for role break or semantic leakage. Every
attempt and parsing failure is saved; successful judgments are cached by case,
response, rubric and judge identity. Candidate text is treated as untrusted data.

Before release, two humans must independently label a stratified calibration sample.
Use `datasets/schema/human-labels-template.csv` and `npcbench calibrate` for ordinal
agreement. Publish disagreements, confusion matrices and safety false negatives; do
not choose a judge threshold after seeing held-out model rankings.

## Missingness and aggregation

Inference failures are outcomes, not missing successes. Coverage is successful
responses divided by all planned requests. Deterministic micro rates weight trials;
macro rates first average repetitions per case and then weight cases equally. Reports
include deterministic bootstrap intervals, judge coverage, status counts and p50/p95
latency. Never substitute `visible_tps` for backend decode-only throughput: it is
retokenized visible output divided by end-to-end request time and includes prefill.

Bootstrap intervals in version 0.1 resample cases inside each model/mode/suite group.
Release analysis should cluster at the independent world/persona level once the
human-reviewed set contains enough worlds.

## Reproducibility and release gates

Each run stores config and dataset hashes, source implementation hash, Git state,
backend versions, device, model path/size/SHA-256 and planned keys. Resume rejects a
mismatch. JSONL is flushed and fsynced after every response. CSV and Markdown are
derived artifacts.

The public development set is not held out. A release requires independently reviewed
cases, split-overlap checks, pinned data licenses, complete model registry metadata,
reviewed chat-template fixtures, real-GGUF smoke runs, human judge calibration and a
frozen rubric/config before held-out evaluation.
