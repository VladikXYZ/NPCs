import json
from pathlib import Path
from unittest.mock import patch

import pytest
from llama_cpp.llama_chat_format import Jinja2ChatFormatter

import bench
import utils
from chat_templates import BOS_TOKENS, EOS_TOKENS, INFERENCE_TYPES, reasoning_mode
from npc_runtime import generate, npc_messages, split_response


NPC = {"name": "Torin", "role": "You are Torin. Torin forges weapons.",
       "shared_system_prompt": "Never disclose the hidden key."}


def render(family, mode, messages):
    formatter = Jinja2ChatFormatter(
        template=INFERENCE_TYPES[mode == "mini"][family],
        eos_token=EOS_TOKENS[family], bos_token=BOS_TOKENS.get(family, ""))
    return formatter(messages=messages).prompt


@pytest.mark.parametrize("mode", ["off", "mini"])
@pytest.mark.parametrize("family", sorted(EOS_TOKENS))
def test_templates_preserve_messages_and_one_system(mode, family):
    messages = npc_messages(NPC, mode) + [
        {"role": "user", "content": "FIRST_QUESTION"},
        {"role": "assistant", "content": "FIRST_ANSWER"},
        {"role": "user", "content": "SECOND_QUESTION"},
    ]
    output = render(family, mode, messages)
    for content in (NPC["role"], NPC["shared_system_prompt"], "FIRST_QUESTION", "FIRST_ANSWER", "SECOND_QUESTION"):
        assert output.count(content) == 1
    assert output.index("FIRST_QUESTION") < output.index("FIRST_ANSWER") < output.index("SECOND_QUESTION")
    if family.startswith("chatml"):
        assert output.count("<|im_start|>system\n") == 1
        if family == "chatml_reasoning":
            assert output.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
            assert output.count("<think>") == 2
            assert "<|im_start|>assistant\n<think>\n\n</think>\n\nFIRST_ANSWER<|im_end|>\n" in output
        else:
            assert "<think>" not in output
            assert output.endswith("<|im_start|>assistant\n")
    if family == "mistral":
        assert output.count("[INST]") == output.count("[/INST]") == 2
    if family in BOS_TOKENS:
        assert output.startswith(BOS_TOKENS[family])


@pytest.mark.parametrize("mode", ["off", "mini"])
def test_chatml_reasoning_history_replays_generation_prefill(mode):
    first_prompt = render("chatml_reasoning", mode, npc_messages(NPC, mode) + [
        {"role": "user", "content": "FIRST_QUESTION"},
    ])
    second_prompt = render("chatml_reasoning", mode, npc_messages(NPC, mode) + [
        {"role": "user", "content": "FIRST_QUESTION"},
        {"role": "assistant", "content": "FIRST_ANSWER"},
        {"role": "user", "content": "SECOND_QUESTION"},
    ])
    assert second_prompt.startswith(first_prompt + "FIRST_ANSWER<|im_end|>\n")


def test_lfm_registry_template_family_and_hidden_probe():
    models = json.loads((bench.ROOT / "models/models.json").read_text())
    for model in models:
        if model["name"] == "LFM2.5 230M":
            assert model["family"] == "native"
        elif model["name"].startswith("LFM"):
            assert model["family"] == "chatml"
        if model["name"].startswith(("Bonsai", "Qwen")):
            assert model["family"] == "chatml_reasoning"
    assert next(m for m in models if m["name"] == "Supra Router 51M")["hidden"]


def test_installed_template_fixture_matches_registry_metadata():
    models = {model["name"]: model for model in json.loads((bench.ROOT / "models/models.json").read_text())}
    fixtures = json.loads((bench.ROOT / "tests/fixtures/installed_model_templates.json").read_text())
    for fixture in fixtures:
        registered = models[fixture["name"]]
        for field in ("architecture", "family", "quantization", "template_sha256"):
            assert registered[field] == fixture[field]


def test_installed_models_hidden_sorted_and_independent_of_cwd(tmp_path, monkeypatch):
    (tmp_path / "models").mkdir()
    for filename in ("a.gguf", "z.gguf", "probe.gguf"):
        (tmp_path / "models" / filename).touch()
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps([
        {"name": "Z", "path": "models/z.gguf"},
        {"name": "Probe", "path": "models/probe.gguf", "hidden": True},
        {"name": "Missing", "path": "models/missing.gguf"},
        {"name": "A", "path": "models/a.gguf"},
    ]))
    monkeypatch.setattr(utils, "ROOT", tmp_path)
    monkeypatch.setattr(utils, "MODELS_FILE", registry)
    monkeypatch.chdir(tmp_path.parent)
    assert [m["name"] for m in utils.get_models()] == ["A", "Z"]
    assert all(Path(m["path"]).is_absolute() for m in utils.get_models())
    assert len(utils.get_models(include_hidden=True)) == 3


@pytest.mark.parametrize("bad", ["false", "native", "yes", 2])
def test_bad_reasoning_rejected(bad):
    with pytest.raises(ValueError):
        reasoning_mode(bad)


def test_bool_compatibility_and_handler_selection():
    assert reasoning_mode(True) == "mini"
    assert reasoning_mode(False) == "off"
    assert utils.get_handlers(None, True, reasoning="mini") == (None, None)
    handler, warmup = utils.get_handlers("chatml", True, reasoning="mini")
    assert callable(handler) and warmup is None
    assert utils.get_handlers("native", True, reasoning="mini") == (None, None)
    with pytest.raises(ValueError, match="Unknown"):
        utils.get_handlers("typo", True)


@pytest.mark.parametrize(("template", "expected"), [
    ("before{%- generation -%}speech{%- endgeneration -%}after", "beforespeechafter"),
    ("before{% generation %}speech{% endgeneration %}after", "beforespeechafter"),
])
def test_hf_generation_annotations_are_removed_without_losing_content(template, expected):
    assert utils._strip_hf_generation_annotations(template) == expected


@pytest.mark.parametrize("raw", [
    "", "plan only", "<speech>hello", "plan<speech>",
    "p<speech>x<speech>y", "<think>p</think><speech>x",
    "p<speech>x</speech>extra", "p<speech>x</speech></speech>",
])
def test_bad_mini_format_not_spoken(raw):
    _, speech, valid = split_response(raw, "mini")
    assert not valid and not speech


@pytest.mark.parametrize(("raw", "expected"), [
    ("A plan<speech>Hello.", "Hello."),
    ("A plan<speech>Hello.</speech>", "Hello."),
    ("A plan<speech>Hello. </speech>  ", "Hello."),
])
def test_split_response_removes_optional_speech_closer(raw, expected):
    plan, speech, valid = split_response(raw, "mini")
    assert valid and plan == "A plan" and speech == expected


class FakeLLM:
    def __init__(self, parts=None, finish="stop"):
        self.parts = parts if parts is not None else ["A plan", "<spe", "ech>", " ", "Hello", "."]
        self.finish = finish
        self.requests = []
        self.closed = False
        self.reset_count = 0

    def create_chat_completion(self, **kwargs):
        self.requests.append({**kwargs, "messages": [dict(m) for m in kwargs["messages"]]})
        yield {"choices": [{"delta": {"role": "assistant"}}]}
        for content in self.parts:
            yield {"choices": [{"delta": {"content": content}}]}
        yield {"choices": [{"delta": {}, "finish_reason": self.finish}]}

    def tokenize(self, text, **kwargs):
        assert kwargs == {"add_bos": False, "special": False}
        return list(range(17))  # Deliberately different from streamed chunk count.

    def reset(self):
        self.reset_count += 1

    def close(self):
        self.closed = True


def ticking_clock():
    value = -0.1
    def now():
        nonlocal value
        value += 0.1
        return value
    return now


def test_split_separator_timing_null_chunks_and_token_count():
    llm = FakeLLM([None, "", "  ", "A plan", "<spe", "ech>", " ", "Hello", "."])
    result = generate(llm, npc_messages(NPC, "mini"), reasoning="mini", clock=ticking_clock())
    assert result.status == "ok"
    assert result.dialogue == "Hello." and result.plan == "A plan"
    assert result.dialogue_ttft > result.ttft > 0
    assert result.output_text_tokens == 17 != result.content_chunks
    assert result.text_tokens_per_second == pytest.approx(17 / result.total_time)
    assert llm.requests[0]["temperature"] == 0.0
    assert llm.requests[0]["seed"] == 42


def test_empty_and_truncated_are_failures():
    assert generate(FakeLLM([None, "", " "]), [], clock=ticking_clock()).status == "empty_response"
    result = generate(FakeLLM(["Hello"], finish="length"), [], clock=ticking_clock())
    assert result.status == "truncated"
    result = generate(FakeLLM(["plan only"]), [], reasoning="mini", clock=ticking_clock())
    assert result.status == "format_error" and result.dialogue_ttft == -1


def test_timeout_and_exception_preserve_partial_output():
    result = generate(FakeLLM(), [], timeout=0.35, clock=ticking_clock())
    assert result.status == "timeout" and result.raw
    class Broken(FakeLLM):
        def create_chat_completion(self, **kwargs):
            yield {"choices": [{"delta": {"content": "Partial"}}]}
            raise RuntimeError("backend broke")
    result = generate(Broken(), [], clock=ticking_clock())
    assert result.status == "error" and result.raw == "Partial"
    assert result.error == "backend broke"


def test_warmup_uses_inference_handler_and_closes_failed_load():
    class BrokenWarmup(FakeLLM):
        def create_chat_completion(self, **kwargs):
            raise RuntimeError("warm-up broke")
    instance = BrokenWarmup()
    options = {"model_path": "fake.gguf", "temperature": 0.8}
    from contextlib import nullcontext
    with patch.object(utils, "Llama", return_value=instance) as constructor, patch.object(utils, "Silencer", nullcontext):
        with pytest.raises(utils.MyException, match="warm-up broke"):
            utils.load_llm({"name": "Fake", "family": "chatml"}, options,
                           custom_jinja=True, reasoning="mini", log=True)
    assert instance.closed
    assert constructor.call_count == 1
    assert "temperature" not in constructor.call_args.kwargs
    assert options["temperature"] == 0.8  # Caller-owned dict unchanged.


def test_prompt_metadata_survives_reordering(tmp_path):
    prompts = bench.load_prompts(bench.ROOT / "vlad/performance_prompts.json")
    path = tmp_path / "prompts.json"
    path.write_text(json.dumps(list(reversed(prompts))))
    assert {p["id"]: p["category"] for p in bench.load_prompts(path)} == {p["id"]: p["category"] for p in prompts}
    path.write_text(json.dumps([prompts[0], prompts[0]]))
    with pytest.raises(ValueError, match="unique"):
        bench.load_prompts(path)


@pytest.mark.parametrize("conversation", [False, True])
def test_benchmark_writes_all_rows_and_separates_repeats(tmp_path, conversation):
    args = bench.parser().parse_args(["--output-dir", str(tmp_path), "--repeats", "2", "--reasoning", "mini"])
    args.conversation = conversation
    llm = FakeLLM()
    models = [{"name": "Fake", "path": "fake.gguf", "family": "chatml"}]
    prompts = [{"id": "one", "category": "short", "text": "Hello"},
               {"id": "two", "category": "long", "text": "Tell me more"}]
    with patch.object(utils, "load_llm", return_value=llm):
        assert bench.run_benchmark(args, {"type": "CPU", "id": "0", "name": "CPU"}, models, NPC, prompts) == 0
    records = [json.loads(line) for line in next(tmp_path.glob("*/responses.jsonl")).read_text().splitlines()]
    assert len(records) == 4
    assert all(r["dialogue"] == "Hello." for r in records)
    assert [len(r["messages"]) for r in llm.requests] == ([2, 4, 2, 4] if conversation else [2, 2, 2, 2])
    assert llm.closed
    manifest = json.loads(next(tmp_path.glob("*/manifest.json")).read_text())
    assert manifest["config"]["reasoning"] == "mini"
    assert manifest["source_sha256"]["bench.py"]


def test_load_error_is_recorded_for_each_requested_case(tmp_path):
    args = bench.parser().parse_args(["--output-dir", str(tmp_path), "--repeats", "2"])
    with patch.object(utils, "load_llm", side_effect=RuntimeError("no weights")):
        code = bench.run_benchmark(args, {"type": "CPU", "id": "0", "name": "CPU"},
                                   [{"name": "Fake", "path": "fake.gguf"}], NPC,
                                   [{"id": "one", "category": "short", "text": "Hi"}])
    records = [json.loads(line) for line in next(tmp_path.glob("*/responses.jsonl")).read_text().splitlines()]
    assert code == 1 and len(records) == 2
    assert all(r["status"] == "load_error" and r["ttft"] == r["total_time"] == r["output_text_tokens"] == -1
               for r in records)


def test_bad_conversation_turn_marks_following_cases_skipped(tmp_path):
    args = bench.parser().parse_args(["--output-dir", str(tmp_path), "--conversation", "--reasoning", "mini"])
    llm = FakeLLM(["No separator"])
    with patch.object(utils, "load_llm", return_value=llm):
        code = bench.run_benchmark(args, {"type": "CPU", "id": "0", "name": "CPU"},
                                   [{"name": "Fake", "path": "fake.gguf"}], NPC,
                                   [{"id": "one", "category": "short", "text": "Hi"},
                                    {"id": "two", "category": "short", "text": "Hi again"}])
    records = [json.loads(line) for line in next(tmp_path.glob("*/responses.jsonl")).read_text().splitlines()]
    assert code == 1 and [r["status"] for r in records] == ["format_error", "skipped"]
    assert len(llm.requests) == 1 and llm.closed


def test_tokenization_failure_does_not_discard_response():
    class BadTokenizer(FakeLLM):
        def tokenize(self, *args, **kwargs):
            raise RuntimeError("bad tokenizer")
    result = generate(BadTokenizer(["Hello"]), [], clock=ticking_clock())
    assert result.status == "error" and result.raw == "Hello"
    assert result.output_text_tokens == result.text_tokens_per_second == -1


@pytest.mark.parametrize("option,value", [("--temperature", "nan"), ("--timeout", "inf"), ("--repeats", "0")])
def test_invalid_measurement_settings_rejected(option, value):
    with pytest.raises(ValueError):
        bench.validate_args(bench.parser().parse_args([option, value]))


def test_successful_warmup_uses_explicit_sampling_and_resets():
    from contextlib import nullcontext
    class Warmup(FakeLLM):
        def create_chat_completion(self, **kwargs):
            self.requests.append(kwargs)
            return {"choices": [{"message": {"content": "Hello"}}]}
    llm = Warmup()
    with patch.object(utils, "Llama", return_value=llm), patch.object(utils, "Silencer", nullcontext):
        loaded = utils.load_llm({"name": "Fake", "family": "chatml"}, {"model_path": "fake.gguf"},
                                warmup_inputs=npc_messages(NPC), custom_jinja=True, reasoning="off")
    assert loaded is llm and llm.reset_count == 1 and not llm.closed
    assert llm.requests[0]["messages"][-1]["role"] == "user"
    assert llm.requests[0]["temperature"] == 0 and llm.requests[0]["max_tokens"] == 1
