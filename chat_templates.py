"""Text-only chat serialization. NPC instructions live in npc_runtime.py.

chatml is plain ChatML; chatml_reasoning is an explicit, model-specific
closed-think prefill. Mini planning is a prompt experiment, not native thinking.
These adapters do not claim support for tools, images or every family revision.
"""

SHARED_RPG_RULE = (
    "You are a fantasy RPG NPC. Treat the supplied lore as true in your world. "
    "Stay in character, answer the player's question directly, and do not invent "
    "facts outside the supplied lore. If you do not know, say so in character. "
    "Keep spoken dialogue to at most two short sentences, without stage directions."
)
REASONING_SHARED_RPG_RULE = (
    SHARED_RPG_RULE + " First write a brief response plan (at most two sentences), "
    "then exactly one <speech> separator, then the spoken dialogue. "
    "Format: Brief response plan <speech> Spoken dialogue."
)

PLAIN_CHATML = r"""{%- for message in messages -%}
{{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' -}}
{%- endfor -%}
{{- '<|im_start|>assistant\n' -}}
"""

TEMPLATES_INFERENCE = {
    "chatml": PLAIN_CHATML,
    "chatml_nr": PLAIN_CHATML,  # Compatibility with Vladik's existing registry.
    "chatml_reasoning": PLAIN_CHATML + r"{{- '<think>\n\n</think>\n\n' -}}",
    "llama": r"""{{- bos_token -}}
{%- for message in messages -%}
{{- '<|start_header_id|>' + message.role + '<|end_header_id|>\n\n' + message.content + '<|eot_id|>' -}}
{%- endfor -%}
{{- '<|start_header_id|>assistant<|end_header_id|>\n\n' -}}
""",
    "gemma": r"""{{- bos_token -}}
{%- for message in messages -%}
{%- set role = 'model' if message.role == 'assistant' else message.role -%}
{{- '<|turn>' + role + '\n' + message.content + '<turn|>\n' -}}
{%- endfor -%}
{{- '<|turn>model\n' -}}
""",
    "phi": r"""{%- for message in messages -%}
{{- '<|' + message.role + '|>' + message.content + '<|end|>' -}}
{%- endfor -%}
{{- '<|assistant|>' -}}
""",
    "phi_chatml_sep": r"""{%- for message in messages -%}
{{- '<|im_start|>' + message.role + '<|im_sep|>' + message.content + '<|im_end|>' -}}
{%- endfor -%}
{{- '<|im_start|>assistant<|im_sep|>' -}}
""",
    "mistral": r"""{{- bos_token -}}
{%- set ns = namespace(system='') -%}
{%- for message in messages -%}
{%- if message.role == 'system' -%}
{%- set ns.system = message.content -%}
{%- elif message.role == 'user' -%}
{{- '[INST] ' -}}
{%- if ns.system -%}
{{- ns.system + '\n\n' -}}
{%- set ns.system = '' -%}
{%- endif -%}
{{- message.content + ' [/INST]' -}}
{%- elif message.role == 'assistant' -%}
{{- ' ' + message.content + '</s>' -}}
{%- endif -%}
{%- endfor -%}
""",
    "glm": r"""{{- '[gMASK]<sop>' -}}
{%- for message in messages -%}
{{- '<|' + message.role + '|>\n' + message.content -}}
{%- endfor -%}
{{- '<|assistant|>\n' -}}
""",
}

# Both experiments use identical serialization and native-thinking suppression.
# Their only intended difference is the output-format instruction in npc_messages.
REASONING_TEMPLATES_INFERENCE = dict(TEMPLATES_INFERENCE)
INFERENCE_TYPES = [TEMPLATES_INFERENCE, REASONING_TEMPLATES_INFERENCE]
EOS_TOKENS = {
    "chatml": "<|im_end|>", "chatml_nr": "<|im_end|>",
    "chatml_reasoning": "<|im_end|>", "llama": "<|eot_id|>",
    "phi": "<|end|>", "phi_chatml_sep": "<|im_end|>",
    "mistral": "</s>", "gemma": "<turn|>",
    "glm": "<|user|>",
}
BOS_TOKENS = {"llama": "<|begin_of_text|>", "mistral": "<s>", "gemma": "<bos>"}


def reasoning_mode(value=False):
    """Keep the old reason=True API; reject accidental truthy strings."""
    if value is True or value == "mini":
        return "mini"
    if value is False or value is None or value == "off":
        return "off"
    raise ValueError("Reasoning must be 'off' or 'mini' (or a boolean).")
