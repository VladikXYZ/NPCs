SHARED_RPG_RULE = "You are a fantasy RPG NPC. Speak ONLY pure dialogue with NO stage directions, actions, or asterisks. Be direct and terse. Answer the player's exact question and immediately stop talking. Do NOT volunteer background facts unless directly asked, and do NOT over-explain. Treat your reality as a normal fantasy world. Maximum length: 2 short sentences."

TEMPLATES = {
    # 1. ChatML (Qwen, Bonsai)
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
    {{- '<|im_start|>assistant\\n<think>\\n\\n</think>\\n' -}}
{%- endif -%}
""".replace("__RULE__", repr(SHARED_RPG_RULE)),

    # 2. Llama 3 / Phi
    "llama_phi": """
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

    # 3. Mistral
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

    # 4. Gemma
    "gemma": """
{%- set shared_prompt = __RULE__ -%}
{%- if messages and messages[0].role == 'system' -%}
    {{- '<start_of_turn>system\\n' + messages[0].content + '\\n\\n' + shared_prompt + '<end_of_turn>\\n' -}}
{%- else -%}
    {{- '<start_of_turn>system\\n' + shared_prompt + '<end_of_turn>\\n' -}}
{%- endif -%}
{%- for message in messages -%}
    {%- if message.role != 'system' -%}
        {{- '<start_of_turn>' + message.role + '\\n' + message.content + '<end_of_turn>\\n' -}}
    {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{- '<start_of_turn>assistant\\n' -}}
{%- endif -%}
""".replace("__RULE__", repr(SHARED_RPG_RULE)),
}

EOS_TOKENS = {
    "qwen": "<|im_end|>",
    "llama_phi": "<|eot_id|>",
    "mistral": "</s>",
    "gemma": "<end_of_turn>",
}

from llama_cpp.llama_chat_format import Jinja2ChatFormatter

def get_handler(model_name: str):
    model_name_lower = model_name.lower()
    if "gemma" in model_name_lower: family = "gemma"
    elif "llama" in model_name_lower or "phi" in model_name_lower: family = "llama_phi"
    elif "mistral" in model_name_lower or "ui-tars" in model_name_lower: family = "mistral"
    elif "bonsai" in model_name_lower or "qwen" in model_name_lower: family = "qwen"
    else: family = "qwen"

    return Jinja2ChatFormatter(
        template=TEMPLATES[family],
        eos_token=EOS_TOKENS[family],
        bos_token=""
    ).to_chat_handler()