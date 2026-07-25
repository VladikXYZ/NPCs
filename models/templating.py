SHARED_RPG_RULE = "You are a fantasy RPG NPC. Speak ONLY pure dialogue with NO stage directions, actions, or asterisks. Be direct and terse. Answer the player's exact question and immediately stop talking. Do NOT volunteer background facts unless directly asked, and do NOT over-explain. Treat your reality as a normal fantasy world. Maximum length: 2 short sentences."

TEMPLATES = {
    # 1. ChatML (Qwen, Bonsai, UI-TARS)
    "qwen": f"""{{% set shared_prompt = {repr(SHARED_RPG_RULE)} %}}
<|im_start|>system
{{{{ shared_prompt }}}}
<|im_end|>
{{% for message in messages %}}
<|im_start|>{{{{ message.role }}}}
{{% if message.role == 'assistant' %}}<think>

</think>

{{% endif %}}{{{{ message.content }}}}
<|im_end|>
{{% endfor %}}
<|im_start|>assistant
<think>

</think>
""",

    # 2. Llama 3 (Llama-3.2-1B-Instruct)
    "llama": f"""{{% set shared_prompt = {repr(SHARED_RPG_RULE)} %}}
<|start_header_id|>system<|end_header_id|>
{{{{ shared_prompt }}}}
<|eot_id|>
{{% for message in messages %}}
<|start_header_id|>{{{{ message.role }}}}<|end_header_id|>
{{{{ message.content }}}}
<|eot_id|>
{{% endfor %}}
<|start_header_id|>assistant<|end_header_id|>
""",

    # 3. Phi-4 (Covers Phi-4-mini-instruct)
    "phi": f"""{{% set shared_prompt = {repr(SHARED_RPG_RULE)} %}}
<|system|>
{{{{ shared_prompt }}}}
<|end|>
{{% for message in messages %}}
<|{{{{ message.role }}}}|>
{{{{ message.content }}}}
<|end|>
{{% endfor %}}
<|assistant|>
""",

    # 4. Mistral (Covers Mistral-7B-Instruct-v0.3)
    "mistral": f"""{{% set shared_prompt = {repr(SHARED_RPG_RULE)} %}}
[INST] {{{{ shared_prompt }}}}
{{% for message in messages %}}
{{% if message.role == 'user' %}}
{{{{ message.content }}}} [/INST]
{{% elif message.role == 'assistant' %}}
{{{{ message.content }}}}</s>
{{% endif %}}
{{% endfor %}}""",

    # 5. Gemma (Covers gemma-4-E2B-it and E4B-it)
    "gemma": f"""{{% set shared_prompt = {repr(SHARED_RPG_RULE)} %}}
<|turn>system
{{{{ shared_prompt }}}}
<turn|>
{{% for message in messages %}}
{{% set role = 'model' if message.role == 'assistant' else message.role %}}
<|turn>{{{{ role }}}}
{{{{ message.content }}}}
<turn|>
{{% endfor %}}
<|turn>model
"""
}

# Ensure your model handler selects the correct EOS token so the benchmark knows when to stop:
EOS_TOKENS = {
    "qwen": "<|im_end|>",
    "llama": "<|eot_id|>",
    "phi": "<|end|>",
    "mistral": "</s>",
    "gemma": "<turn|>",
}

from llama_cpp.llama_chat_format import Jinja2ChatFormatter


def get_handler(model_name: str):
    # return None
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