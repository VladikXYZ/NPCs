from llama_cpp.llama_chat_format import Jinja2ChatFormatter

SHARED_RPG_RULE = "You are a fantasy RPG NPC. Speak ONLY pure dialogue with NO stage directions, actions, or asterisks. Be direct and terse. Answer the player's exact question and immediately stop talking. Do NOT volunteer background facts unless directly asked, and do NOT over-explain. Treat your reality as a normal fantasy world. Maximum length: 2 short sentences."

# --- INFERENCE TEMPLATES (Appends generation prompt) ---
TEMPLATES_INFERENCE = {
    "qwen": """{%- set shared_prompt = __RULE__ -%}
{{- '<|im_start|>system\\n' + shared_prompt + '<|im_end|>\\n' -}}
{%- for message in messages -%}
<|im_start|>{{ message.role }}
{{ message.content }}<|im_end|>
{% endfor -%}
{{- '<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n' -}}
""".replace("__RULE__", repr(SHARED_RPG_RULE)),

    "llama": """{%- set shared_prompt = __RULE__ -%}
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
{{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' -}}
""".replace("__RULE__", repr(SHARED_RPG_RULE)),

    "phi": """{%- set shared_prompt = __RULE__ -%}
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
{{- '<|assistant|>' -}}
""".replace("__RULE__", repr(SHARED_RPG_RULE)),

    "mistral": """{%- set shared_prompt = __RULE__ -%}
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

    "gemma": """{%- set shared_prompt = __RULE__ -%}
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
{{- '<|turn>model\\n' -}}
""".replace("__RULE__", repr(SHARED_RPG_RULE)),
}

# --- WARMUP TEMPLATES (No generation prompt at the end) ---
QWEN_WARMUP = """{%- set shared_prompt = __RULE__ -%}
{{- '<|im_start|>system\\n' + shared_prompt + '<|im_end|>\\n' -}}
{%- for message in messages -%}
<|im_start|>{{ message.role }}
{{ message.content }}<|im_end|>
{% endfor -%}
""".replace("__RULE__", repr(SHARED_RPG_RULE))

EOS_TOKENS = {
    "qwen": "<|im_end|>",
    "llama": "<|eot_id|>",
    "phi": "<|end|>",
    "mistral": "</s>",
    "gemma": "<turn|>",
}

def get_handlers(model_name: str):
    """Returns a tuple: (handler_inference, handler_warmup)"""
    model_name_lower = model_name.lower()

    if "gemma" in model_name_lower:
        family = "gemma"
    elif "phi" in model_name_lower:
        family = "phi"
    elif "llama" in model_name_lower:
        family = "llama"
    elif "mistral" in model_name_lower:
        family = "mistral"
    else:
        family = "qwen"

    handler_inference = Jinja2ChatFormatter(
        template=TEMPLATES_INFERENCE[family],
        eos_token=EOS_TOKENS[family],
        bos_token=""
    ).to_chat_handler()

    if family == "qwen":
        handler_warmup = Jinja2ChatFormatter(
            template=QWEN_WARMUP,
            eos_token=EOS_TOKENS[family],
            bos_token=""
        ).to_chat_handler()

        return handler_inference, handler_warmup
    return handler_inference, None