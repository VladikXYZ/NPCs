# Dataset provenance

`datasets/dev/cases.jsonl` is a development fixture authored for this repository. It is intentionally public, must not be used as a hidden test set, and each case carries a review-status warning. Before a release, two reviewers should independently inspect every target, private fact, canary, and ambiguity.

`chars.json` is a legacy conversation file referenced by `trash.py` as originating from `chimbiwide/NPC-Dialogue_v2`. That attribution has not yet been pinned to a dataset revision or license in this repository, so the file is excluded from benchmark scoring. Run `python -m npcbench audit-data chars.json` to reproduce its structural audit.

`jakub/generated_prompts.json` and `martin/jailbreak_template.json` are legacy development materials. They can be converted with `npcbench migrate`, but converted cases remain development data and retain source metadata.

