SHARED_RPG_RULE = "You are a fantasy RPG NPC. Speak ONLY pure dialogue with NO stage directions, actions, or asterisks. Be direct and terse. Answer the player's exact question and immediately stop talking. Do NOT volunteer background facts unless directly asked, and do NOT over-explain. Treat your reality as a normal fantasy world. Maximum length: 2 short sentences."

TEMPLATES_INFERENCE = {
    "chatml": """{%- set shared_prompt = __RULE__ -%}
{{- '<|im_start|>system\\n' + shared_prompt + '<|im_end|>\\n' -}}
{%- for message in messages -%}
<|im_start|>{{ message.role }}
{{ message.content }}<|im_end|>
{% endfor -%}
{{- '<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n' -}}
""".replace("__RULE__", repr(SHARED_RPG_RULE)),

    "llama": """{%- set shared_prompt = __RULE__ -%}
{{- '<|start_header_id|>system<|end_header_id|>\\n\\n' -}}
{%- if messages and messages[0].role == 'system' -%}
    {{- messages[0].content + '\\n\\n' -}}
{%- endif -%}
{{- shared_prompt + '<|eot_id|>' -}}
{%- for message in messages -%}
    {%- if message.role != 'system' -%}
        {{- '<|start_header_id|>' + message.role + '<|end_header_id|>\\n\\n' + message.content + '<|eot_id|>' -}}
    {%- endif -%}
{%- endfor -%}
{{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' -}}
""".replace("__RULE__", repr(SHARED_RPG_RULE)),

    "gemma": """{%- set shared_prompt = __RULE__ -%}
{{- '<|turn>system\\n' -}}
{%- if messages and messages[0].role == 'system' -%}
    {{- messages[0].content + '\\n\\n' -}}
{%- endif -%}
{{- shared_prompt + '<turn|>\\n' -}}
{%- for message in messages -%}
    {%- if message.role != 'system' -%}
        {%- set role = 'model' if message.role == 'assistant' else message.role -%}
        {{- '<|turn>' + role + '\\n' + message.content + '<turn|>\\n' -}}
    {%- endif -%}
{%- endfor -%}
{{- '<|turn>model\\n<|channel>thought\\n<channel|>' -}}
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

    "glm": """{%- set shared_prompt = __RULE__ -%}
[gMASK]<sop><|system|>\\n
{%- if messages and messages[0].role == 'system' -%}
    {{- messages[0].content + '\\n\\n' -}}
{%- endif -%}
{{- shared_prompt -}}
{%- for message in messages -%}
    {%- if message.role != 'system' -%}
        {{- '\\n<|' + message.role + '|>\\n' + message.content -}}
    {%- endif -%}
{%- endfor -%}
{{- '\\n<|assistant|>\\n' -}}
""".replace("__RULE__", repr(SHARED_RPG_RULE)),
}

QWEN_WARMUP = """{%- set shared_prompt = __RULE__ -%}
{{- '<|im_start|>system\\n' + shared_prompt + '<|im_end|>\\n' -}}
{%- for message in messages -%}
<|im_start|>{{ message.role }}
{{ message.content }}<|im_end|>
{% endfor -%}
""".replace("__RULE__", repr(SHARED_RPG_RULE))

EOS_TOKENS = {
    "chatml": "<|im_end|>",
    "llama": "<|eot_id|>",
    "phi": "<|end|>",
    "mistral": "</s>",
    "gemma": "<turn|>",
    "glm": "<|user|>",
}