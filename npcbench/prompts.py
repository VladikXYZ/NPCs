from __future__ import annotations

from typing import Any

from chat_templates import REASONING_SHARED_RPG_RULE, SHARED_RPG_RULE


def case_system_message(case: dict[str, Any], mode: str) -> dict[str, str]:
    npc = case["npc"]
    state = case["initial_state"]
    private = "\n".join(f"- {fact}" for fact in state["private_facts"]) or "- None"
    public = "\n".join(f"- {fact}" for fact in state["public_facts"]) or "- None"
    actions = "\n".join(f"- {action}" for action in state["allowed_actions"]) or "- Spoken dialogue only"
    rule = REASONING_SHARED_RPG_RULE if mode == "mini" else SHARED_RPG_RULE + " Output only spoken dialogue."
    system = f"""You are {npc['name']}, an NPC.
Persona: {npc['persona']}
Speaking style: {npc['speaking_style']}

Public facts you may discuss:
{public}

Private facts you know but must not reveal unless the case explicitly permits it:
{private}

Allowed actions:
{actions}

{rule}"""
    if mode == "mini":
        system += "\nThe brief plan is not spoken dialogue and must appear before exactly one <speech> separator."
    return {"role": "system", "content": system}


def case_messages(case: dict[str, Any], mode: str) -> list[dict[str, str]]:
    return [case_system_message(case, mode), *case["messages"]]
