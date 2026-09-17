"""Shared NPC prompting and observable streaming measurements."""

from dataclasses import asdict, dataclass
import time

from chat_templates import SHARED_RPG_RULE, REASONING_SHARED_RPG_RULE, reasoning_mode


def npc_messages(npc, reasoning="off"):
    mode = reasoning_mode(reasoning)
    rule = REASONING_SHARED_RPG_RULE if mode == "mini" else SHARED_RPG_RULE + " Output only spoken dialogue."
    # Scope existing dialogue-only rules to speech, avoiding a contradictory
    # instruction when the experimental mode also requests a plan.
    parts = [npc["role"], npc.get("shared_system_prompt", ""), rule]
    if mode == "mini":
        parts.append("The brief plan is a separate output field. Earlier dialogue-only "
                     "and length instructions apply to the text after <speech>.")
    return [{"role": "system", "content": "\n\n".join(p for p in parts if p)}]


def split_response(raw, reasoning="off"):
    """Return (plan, dialogue, format_ok); never pass malformed plans as speech."""
    if reasoning_mode(reasoning) == "off":
        valid = bool(raw.strip()) and not any(tag in raw for tag in ("<think>", "</think>", "<speech>"))
        return "", raw.strip() if valid else "", valid
    if raw.count("<speech>") != 1:
        return "", "", False
    plan, dialogue = (part.strip() for part in raw.split("<speech>", 1))
    valid = bool(plan and dialogue) and not any(tag in raw for tag in ("<think>", "</think>"))
    return plan, dialogue if valid else "", valid


@dataclass
class Generation:
    raw: str = ""
    plan: str = ""
    dialogue: str = ""
    status: str = "ok"
    error: str = ""
    ttft: float = -1
    dialogue_ttft: float = -1
    total_time: float = -1
    output_text_tokens: int = -1
    text_tokens_per_second: float = -1
    finish_reason: str = ""
    content_chunks: int = 0

    def as_dict(self):
        return asdict(self)


def generate(llm, messages, *, reasoning="off", max_tokens=256, temperature=0.0,
             seed=42, timeout=60.0, on_text=None, clock=time.perf_counter):
    """TTFT is first non-whitespace text; text throughput includes prefill.

    Chunks are NOT model tokens. Retokenization measures output text only,
    excluding hidden/EOS tokens, and is not native decode throughput.
    Timeout is cooperative: checked when the backend yields; a stalled native
    call needs the subprocess watchdog described in the research plan.
    """
    mode = reasoning_mode(reasoning)
    if max_tokens <= 0 or timeout <= 0:
        raise ValueError("max_tokens and timeout must be positive")
    result = Generation(output_text_tokens=0)
    start = clock()
    stream = None
    try:
        stream = llm.create_chat_completion(messages=messages, stream=True,
                                           max_tokens=max_tokens, temperature=temperature, seed=seed)
        for chunk in stream:
            elapsed = clock() - start
            if elapsed > timeout:
                result.status, result.error = "timeout", "Cooperative request timeout exceeded."
                break
            choices = chunk.get("choices", [])
            if not choices:
                continue
            choice = choices[0]
            if choice.get("finish_reason"):
                result.finish_reason = choice["finish_reason"]
            content = choice.get("delta", {}).get("content")
            if not isinstance(content, str) or not content:
                continue
            result.content_chunks += 1
            result.raw += content
            if result.ttft < 0 and result.raw.strip():
                result.ttft = elapsed
            _, speech, valid = split_response(result.raw, mode)
            if valid and speech and result.dialogue_ttft < 0:
                result.dialogue_ttft = elapsed
            if on_text:
                on_text(content)
    except Exception as exc:
        result.status, result.error = "error", str(exc)
    finally:
        if stream is not None and hasattr(stream, "close"):
            try:
                stream.close()
            except Exception as exc:
                result.status, result.error = "error", f"Stream cleanup failed: {exc}"
        result.total_time = clock() - start

    result.plan, result.dialogue, valid = split_response(result.raw, mode)
    if result.status == "ok":
        if not result.raw.strip():
            result.status = "empty_response"
        elif result.finish_reason == "length":
            result.status = "truncated"
        elif not valid:
            result.status = "format_error"
        if result.total_time > timeout:
            result.status = "timeout"
    if not valid:
        result.dialogue_ttft = -1
    if result.raw:
        try:
            result.output_text_tokens = len(llm.tokenize(result.raw.encode("utf-8"), add_bos=False, special=False))
            if result.total_time > 0:
                result.text_tokens_per_second = result.output_text_tokens / result.total_time
        except Exception as exc:
            result.output_text_tokens = -1
            result.status = "error"
            result.error = (result.error + f" Output tokenization failed: {exc}").strip()
    return result
