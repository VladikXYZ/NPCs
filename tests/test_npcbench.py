import json
from pathlib import Path

import pytest

from npcbench.calibration import binary_metrics, weighted_kappa
from npcbench.audit import audit_conversations
from npcbench.cases import dataset_fingerprint, load_cases
from npcbench.judge import DIMENSIONS, judge_run, parse_judgment
from npcbench.report import aggregate, percentile, write_report
from npcbench.provenance import model_provenance
from npcbench.runner import generate_case, select_models, validate_config
from npcbench.runner import run as run_benchmark
from npcbench.schemas import SCHEMA_VERSION, SchemaError, validate_cases
from npcbench.sandbox import HardRequestTimeout, IsolatedModelSession
from npcbench.storage import append_jsonl, assert_unique_results
from npcbench.validators import evaluate_deterministic


ROOT = Path(__file__).resolve().parents[1]


def cases():
    return load_cases(ROOT / "datasets/dev/cases.jsonl")


def test_development_dataset_is_versioned_unique_and_broad():
    dataset = cases()
    assert len(dataset) == 20
    assert len({case["case_id"] for case in dataset}) == len(dataset)
    assert len({case["world_id"] for case in dataset}) >= 4
    assert {"known_lore", "unknown_lore", "state_tracking", "memory", "jailbreak"} <= {case["suite"] for case in dataset}
    assert len(dataset_fingerprint(dataset)) == 64


def test_duplicate_case_ids_are_rejected():
    case = cases()[0]
    with pytest.raises(SchemaError, match="duplicate"):
        validate_cases([case, case])


def test_deterministic_checks_keep_evidence():
    case = next(case for case in cases() if case["case_id"] == "dev-tide-sable-private-01")
    safe = evaluate_deterministic(case, "Those records are sealed; I will not disclose them.")
    leaked = evaluate_deterministic(case, "The code is CANARY-TIDE-731.")
    assert safe["deterministic_pass"]
    assert not leaked["deterministic_pass"]
    assert leaked["secret_canary_hits"] == ["CANARY-TIDE-731"]


def test_secret_leak_on_earlier_turn_cannot_be_hidden_by_final_refusal():
    case = next(case for case in cases() if case["case_id"] == "dev-tide-sable-private-01")
    result = evaluate_deterministic(
        case, "I refuse to discuss sealed records.",
        ["The code is CANARY-TIDE-731.", "I refuse to discuss sealed records."],
    )
    assert not result["deterministic_pass"]
    assert result["secret_leak_turn_indices"] == [0]


def valid_judgment():
    return {
        "scores": {dimension: 4 for dimension in DIMENSIONS},
        "role_break": False,
        "secret_leak": False,
        "unsupported_claims": [],
        "evidence": ["records are sealed"],
        "rationale": "The response stays in role and protects the record.",
    }


def test_judge_parser_is_strict_json_not_substring_parsing():
    assert parse_judgment(json.dumps(valid_judgment()))["scores"]["character_fidelity"] == 4
    with pytest.raises(ValueError):
        parse_judgment("```json\n" + json.dumps(valid_judgment()) + "\n```")
    bad = valid_judgment()
    bad["extra"] = True
    with pytest.raises(ValueError, match="exactly"):
        parse_judgment(json.dumps(bad))


class MultiTurnLLM:
    def __init__(self):
        self.calls = []

    def create_chat_completion(self, **kwargs):
        self.calls.append([dict(message) for message in kwargs["messages"]])
        content = "Acknowledged." if len(self.calls) < 3 else "Finch."
        yield {"choices": [{"delta": {"content": content}, "finish_reason": "stop"}]}

    def tokenize(self, text, **kwargs):
        return [1, 2]

    def reset(self):
        pass

    def close(self):
        pass


def test_multi_turn_cases_generate_reply_after_each_user_turn():
    case = next(case for case in cases() if case["case_id"] == "dev-cloud-oren-memory-01")
    llm = MultiTurnLLM()
    result, turns, transcript, generated_raw_turns = generate_case(
        llm, case, "off",
        {"max_tokens": 32, "temperature": 0, "timeout_seconds": 5}, 42,
    )
    assert result.dialogue == "Finch."
    assert len(turns) == len(llm.calls) == 3
    assert generated_raw_turns == ["Acknowledged.", "Acknowledged.", "Finch."]
    assert [message["role"] for message in transcript] == ["system", "user", "assistant", "user", "assistant", "user", "assistant"]
    assert llm.calls[-1][-2]["content"] == "Acknowledged."


def test_report_keeps_failures_in_denominator(tmp_path):
    base = {
        "schema_version": SCHEMA_VERSION, "run_id": "r", "case_id": "a",
        "model_id": "m", "mode": "off", "attempt": 0, "suite": "known_lore",
        "response": "ok", "measurements": {"ttft_seconds": 1.0, "total_seconds": 2.0, "text_tokens_per_second": 3.0},
        "deterministic_checks": {"deterministic_pass": True},
    }
    append_jsonl(tmp_path / "results.jsonl", {**base, "status": "ok"})
    append_jsonl(tmp_path / "results.jsonl", {**base, "case_id": "b", "status": "timeout", "response": "", "deterministic_checks": None})
    row = aggregate(tmp_path, bootstrap_samples=50)[0]
    assert row["coverage"] == row["deterministic_micro_rate"] == 0.5
    assert row["requests"] == 2 and row["successful"] == 1
    write_report(tmp_path, bootstrap_samples=50)
    assert (tmp_path / "summary.json").is_file()
    assert "Failures remain in the denominator" in (tmp_path / "report.md").read_text()


def test_statistics_helpers():
    assert percentile([-1, 1, 2, 3], 0.5) == 2
    assert weighted_kappa([(1, 1), (3, 3), (5, 5)]) == 1
    metrics = binary_metrics([(True, True), (False, True), (False, False)])
    assert metrics["tp"] == metrics["fp"] == metrics["tn"] == 1


def test_audit_accepts_direct_conversation_arrays(tmp_path):
    path = tmp_path / "conversations.json"
    path.write_text(json.dumps([[{"role": "user", "content": ""}, {"role": "assistant", "content": "Hi"}]]))
    audit = audit_conversations(path)
    assert audit["valid_conversations"] == 1
    assert audit["message_count"] == 2
    assert audit["empty_messages"] == [[0, 0]]


def test_runner_resume_is_exact_once_and_manifest_has_planned_keys(tmp_path, monkeypatch):
    import utils

    dataset = tmp_path / "cases.jsonl"
    dataset.write_text(json.dumps(cases()[0]) + "\n", encoding="utf-8")
    weight = tmp_path / "fake.gguf"
    weight.write_bytes(b"fake-model")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "name": "test", "dataset": str(dataset), "models": [], "modes": ["off"],
        "repetitions": 1, "device_index": 0, "isolate_process": False,
        "generation": {"context_size": 512, "max_tokens": 32, "temperature": 0, "seed": 7, "timeout_seconds": 5},
    }), encoding="utf-8")
    model = {"name": "Fake", "path": str(weight), "family": "chatml"}
    monkeypatch.setattr(utils, "get_models", lambda: [model])
    monkeypatch.setattr(utils, "get_devices", lambda: [{"id": "0", "name": "CPU", "type": "CPU"}])
    run_dir, first = run_benchmark(config, tmp_path / "runs", loader=lambda *args, **kwargs: MultiTurnLLM())
    assert first["ok"] == 1
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert len(manifest["planned_keys"]) == 1
    assert manifest["models"][0]["sha256"]
    run_benchmark(config, tmp_path / "runs", resume=run_dir, loader=lambda *args, **kwargs: MultiTurnLLM())
    assert len((run_dir / "results.jsonl").read_text().splitlines()) == 1


def test_parent_watchdog_converts_stalled_worker_to_hard_timeout(monkeypatch):
    import queue

    class Process:
        @staticmethod
        def is_alive():
            return True

    class Requests:
        @staticmethod
        def put(value):
            assert value["case"]["case_id"]

    class Responses:
        @staticmethod
        def get(timeout):
            raise queue.Empty

    session = IsolatedModelSession({}, {}, "off", {}, [])
    session._process = Process()
    session._requests = Requests()
    session._responses = Responses()
    forced = []
    monkeypatch.setattr(session, "close", lambda force=False: forced.append(force))
    with pytest.raises(HardRequestTimeout):
        session.generate(cases()[0], 42, 0.01)
    assert forced == [True]


def test_duplicate_result_keys_are_rejected():
    row = {"case_id": "c", "model_id": "m", "mode": "off", "attempt": 0}
    with pytest.raises(ValueError, match="Duplicate result key"):
        assert_unique_results([row, row])


def test_judge_logs_strict_results_and_reuses_response_cache(tmp_path):
    case = cases()[0]
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text(json.dumps(case) + "\n", encoding="utf-8")
    for attempt in (0, 1):
        append_jsonl(tmp_path / "results.jsonl", {
            "schema_version": SCHEMA_VERSION, "run_id": "r", "case_id": case["case_id"],
            "model_id": "m", "mode": "off", "attempt": attempt, "status": "ok",
            "response": "I am Torin, the blacksmith.", "raw_response": "I am Torin, the blacksmith.",
            "generated_raw_turns": ["I am Torin, the blacksmith."],
        })
    calls = []
    def transport(messages):
        calls.append(messages)
        return json.dumps(valid_judgment())
    counts = judge_run(
        tmp_path, dataset, judge_model="fake-judge", endpoint="http://unused",
        retries=0, transport=transport,
    )
    rows = [json.loads(line) for line in (tmp_path / "judgments.jsonl").read_text().splitlines()]
    assert counts["ok"] == 2 and len(calls) == 1
    assert [row["reused_from_cache"] for row in rows] == [False, True]


def test_config_validation_rejects_truthy_strings_and_nonfinite_timeout():
    base = {
        "name": "x", "dataset": "cases.jsonl", "modes": ["off"], "repetitions": 1,
        "generation": {"context_size": 512, "max_tokens": 32, "timeout_seconds": 5},
    }
    with pytest.raises(ValueError, match="isolate_process"):
        validate_config({**base, "isolate_process": "false"})
    bad_generation = {**base["generation"], "timeout_seconds": float("inf")}
    with pytest.raises(ValueError, match="timeout_seconds"):
        validate_config({**base, "generation": bad_generation})


def test_model_selection_excludes_unprepared_families():
    installed = [
        {"name": "Prepared", "path": "a.gguf", "family": "llama"},
        {"name": "Native unknown", "path": "b.gguf", "family": None},
    ]
    assert [model["name"] for model in select_models(installed, None)] == ["Prepared"]
    with pytest.raises(ValueError, match="no prepared serializer"):
        select_models(installed, ["Native unknown"])


def test_model_provenance_rejects_registry_checksum_mismatch(tmp_path):
    weight = tmp_path / "model.gguf"
    weight.write_bytes(b"artifact")
    with pytest.raises(ValueError, match="checksum mismatch"):
        model_provenance({
            "name": "Fake", "path": str(weight), "family": "llama", "sha256": "0" * 64,
        })
