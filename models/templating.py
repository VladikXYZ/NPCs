SHARED_RPG_RULE = "You are a fantasy RPG NPC. Speak ONLY pure dialogue with NO stage directions, actions, or asterisks. Be direct and terse. Answer the player's exact question and immediately stop talking. Do NOT volunteer background facts unless directly asked, and do NOT over-explain. Treat your reality as a normal fantasy world. Maximum length: 2 short sentences."

TEMPLATES = {
    # 1. ChatML (Qwen, Bonsai, UI-TARS)
    "qwen": """
{%- set shared_prompt = __RULE__ -%}
{%- if messages and messages[0].role == 'system' -%}
    {{- '<|im_start|>system\\n' + messages[0].content + '\\n\\n' + shared_prompt + '<|im_end|>\\n' -}}
{%- else -%}
    {{- '<|im_start|>system\\n' + shared_prompt + '<|im_end|>\\n' -}}
{%- endif -%}
{%- for message in messages -%}
    {%- if message.role != 'system' -%}
        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' -}}
    {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{- '<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n' -}}
{%- endif -%}
""".replace("__RULE__", repr(SHARED_RPG_RULE)),

    # 2. Llama 3 (Llama-3.2-1B-Instruct)
    "llama": """
{%- set shared_prompt = __RULE__ -%}
{%- if messages and messages[0].role == 'system' -%}
    {{- '<|start_header_id|>system<|end_header_id|>\\n\\n' + messages[0].content + '\\n\\n' + shared_prompt + '<|eot_id|>' -}}
{%- else -%}
    {{- '<|start_header_id|>system<|end_header_id|>\\n\\n' + shared_prompt + '<|eot_id|>' -}}
{%- endif -%}
{%- for message in messages -%}
    {%- if message.role != 'system' -%}
        {{- '<|start_header_id|>' + message.role + '<|end_header_id|>\\n\\n' + message.content + '<|eot_id|>' -}}
    {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' -}}
{%- endif -%}
""".replace("__RULE__", repr(SHARED_RPG_RULE)),

    # 3. Phi-4 (Phi-4-mini-instruct)
    "phi": """
{%- set shared_prompt = __RULE__ -%}
{%- if messages and messages[0].role == 'system' -%}
    {{- '<|system|>' + messages[0].content + '\\n\\n' + shared_prompt + '<|end|>' -}}
{%- else -%}
    {{- '<|system|>' + shared_prompt + '<|end|>' -}}
{%- endif -%}
{%- for message in messages -%}
    {%- if message.role != 'system' -%}
        {{- '<|' + message.role + '|>' + message.content + '<|end|>' -}}
    {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{- '<|assistant|>' -}}
{%- endif -%}
""".replace("__RULE__", repr(SHARED_RPG_RULE)),

    # 4. Mistral (Mistral-7B-Instruct-v0.3)
    "mistral": """
{%- set shared_prompt = __RULE__ -%}
{%- for message in messages -%}
    {%- if message.role == 'system' -%}
        {{- '[INST] ' + message.content + '\\n\\n' + shared_prompt + '\\n\\n' -}}
    {%- elif message.role == 'user' -%}
        {{- message.content + ' [/INST]' -}}
    {%- elif message.role == 'assistant' -%}
        {{- ' ' + message.content + '</s>' -}}
    {%- endif -%}
{%- endfor -%}
""".replace("__RULE__", repr(SHARED_RPG_RULE)),

    # 5. Gemma 4 (gemma-4-E2B/E4B-it)
    "gemma": """
{%- set shared_prompt = __RULE__ -%}
{%- if messages and messages[0].role == 'system' -%}
    {{- '<|turn>system\\n' + messages[0].content + '\\n\\n' + shared_prompt + '<turn|>\\n' -}}
{%- else -%}
    {{- '<|turn>system\\n' + shared_prompt + '<turn|>\\n' -}}
{%- endif -%}
{%- for message in messages -%}
    {%- if message.role != 'system' -%}
        {%- set role = 'model' if message.role == 'assistant' else message.role -%}
        {{- '<|turn>' + role + '\\n' + message.content + '<turn|>\\n' -}}
    {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{- '<|turn>model\\n' -}}
{%- endif -%}
""".replace("__RULE__", repr(SHARED_RPG_RULE)),
}

EOS_TOKENS = {
    "qwen": "<|im_end|>",
    "llama": "<|eot_id|>",
    "phi": "<|end|>",
    "mistral": "</s>",
    "gemma": "<turn|>",
}

from llama_cpp.llama_chat_format import Jinja2ChatFormatter


def get_handler(model_name: str):
    return None
    model_name_lower = model_name.lower()

    if "gemma" in model_name_lower:
        family = "gemma"
    elif "phi" in model_name_lower:
        family = "phi"
    elif "llama" in model_name_lower:
        family = "llama"
    elif "mistral" in model_name_lower:
        family = "mistral"
    # elif "bonsai" in model_name_lower or "qwen" in model_name_lower:
    #     family = "qwen"
    else:
        family = "qwen"

    return Jinja2ChatFormatter(
        template=TEMPLATES[family],
        eos_token=EOS_TOKENS[family],
        bos_token=""
    ).to_chat_handler()