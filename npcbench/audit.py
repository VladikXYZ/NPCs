from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .schemas import fingerprint


def audit_conversations(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Conversation dataset must be a JSON array.")
    empty = []
    malformed = []
    role_counts: Counter[str] = Counter()
    lengths: list[int] = []
    hashes: Counter[str] = Counter()
    for conversation_index, conversation in enumerate(payload):
        if isinstance(conversation, list):
            messages = conversation
        elif isinstance(conversation, dict):
            messages = conversation.get("conversations", conversation.get("messages"))
        else:
            messages = None
        if not isinstance(messages, list):
            malformed.append(conversation_index)
            continue
        lengths.append(len(messages))
        hashes[fingerprint(messages)] += 1
        for message_index, message in enumerate(messages):
            if not isinstance(message, dict):
                malformed.append([conversation_index, message_index])
                continue
            role = str(message.get("role", message.get("from", "missing")))
            role_counts[role] += 1
            content = message.get("content", message.get("value", ""))
            if not isinstance(content, str) or not content.strip():
                empty.append([conversation_index, message_index])
    return {
        "path": str(source),
        "sha256": __import__("hashlib").sha256(source.read_bytes()).hexdigest(),
        "conversations": len(payload),
        "valid_conversations": len(lengths),
        "message_count": sum(lengths),
        "min_messages": min(lengths) if lengths else None,
        "max_messages": max(lengths) if lengths else None,
        "role_counts": dict(sorted(role_counts.items())),
        "empty_messages": empty,
        "malformed": malformed,
        "duplicate_conversation_groups": sum(count > 1 for count in hashes.values()),
        "duplicate_conversations_beyond_first": sum(max(0, count - 1) for count in hashes.values()),
    }
