import json
import os
import subprocess
import sys
import re
import tempfile
from contextlib import contextmanager
from llama_cpp.llama_chat_format import Jinja2ChatFormatter

from vlad.tokens_counter import message

DEVICES_FILE = "devices.json"
MODELS_FILE = "models/models.json"
MODELS_DIRECTORY = "models"

class OutOfTimeError(Exception):
    def __init__(self):
        super().__init__("Ran out of time.")

class LoadError(Exception):
    def __init__(self, mesage):
        super().__init__(message)

class InferenceError(Exception):
    def __init__(self, mesage):
        super().__init__(message)

@contextmanager
def Silencer(suppress=True):
    if suppress:
        old_stderr = os.dup(sys.stderr.fileno())
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stderr.fileno())
        try: yield
        finally:
            os.dup2(old_stderr, sys.stderr.fileno())
            os.close(old_stderr)
            os.close(devnull)
    else: yield

@contextmanager
def Catcher():
    sys.stdout.flush()
    sys.stderr.flush()
    fd_out = sys.stdout.fileno()
    fd_err = sys.stderr.fileno()
    
    old_out = os.dup(fd_out)
    old_err = os.dup(fd_err)
    
    with tempfile.TemporaryFile() as tmp:
        os.dup2(tmp.fileno(), fd_out)
        os.dup2(tmp.fileno(), fd_err)

        logs = [""]
        try: yield logs
        finally:
            sys.stdout.flush()
            sys.stderr.flush()

            os.dup2(old_out, fd_out)
            os.dup2(old_err, fd_err)
            os.close(old_out)
            os.close(old_err)

            tmp.seek(0)
            logs[0] = tmp.read().decode('utf-8', errors='replace')


def _find_device():
    print("🔍 Scanning hardware... (this takes a second)")
    script = f"""
import sys
from llama_cpp import Llama
try:
   llm = Llama(model_path='models/Supra-Router-51M-Q1_0.gguf', n_gpu_layers=1, verbose=True)
except Exception:
   pass
    """
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, encoding='utf-8')
    devices = []
    for line in result.stderr.split('\n'):
        match = re.search(r"ggml_vulkan:\s+(\d+)\s+=\s+(.*?)\s+\|", line)
        if match:
            devices.append({"id": match.group(1), "name": match.group(2).strip(), "type": "Vulkan"})

    cpu_id = str(len(devices))
    devices.append({"id": cpu_id, "name": "CPU", "type": "CPU"})

    with open(DEVICES_FILE, "w") as f:
        json.dump(devices, f)
    return devices

def get_devices():
    try:
        with open(DEVICES_FILE, "r") as f:
            devices = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        devices = _find_device()
    return devices

def get_models():
    with open(MODELS_FILE, "r") as f:
        return json.load(f)

SHARED_RPG_RULE = "You are a fantasy RPG NPC. Speak ONLY pure dialogue with NO stage directions, actions, or asterisks. Be direct and terse. Answer the player's exact question and immediately stop talking. Do NOT volunteer background facts unless directly asked, and do NOT over-explain. Treat your reality as a normal fantasy world. Maximum length: 2 short sentences."

TEMPLATES_INFERENCE = {
    # ChatML: Qwen, LFM, Bonsai, Jan, Falcon-Mamba, DiffuCoder, UI-TARS, WeDLM
    "chatml": """{%- set shared_prompt = __RULE__ -%}
{{- '<|im_start|>system\\n' -}}
{%- if messages and messages[0].role == 'system' -%}
    {{- messages[0].content + '\\n\\n' -}}
{%- endif -%}
{{- shared_prompt + '<|im_end|>\\n' -}}
{%- for message in messages -%}
    {%- if message.role != 'system' -%}
        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' -}}
    {%- endif -%}
{%- endfor -%}
{{- '<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n' -}}
""".replace("__RULE__", repr(SHARED_RPG_RULE)),

    # Llama 3 / 3.1 / 3.2
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

    # Gemma 4
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

    # Phi 4
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

    # Mistral / Instruct
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

    # GLM-4 / GLM-4.7
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

def get_handlers(family: str):
    if not family: return None, None

    handler_inference = Jinja2ChatFormatter(
        template=TEMPLATES_INFERENCE[family],
        eos_token=EOS_TOKENS[family],
        bos_token=""
    ).to_chat_handler()

    if family == "chatml":
        handler_warmup = Jinja2ChatFormatter(
            template=QWEN_WARMUP,
            eos_token=EOS_TOKENS[family],
            bos_token=""
        ).to_chat_handler()

        return handler_inference, handler_warmup
    return handler_inference, None

if __name__ == '__main__':
    models = sorted([os.path.splitext(os.path.basename(x))[0] for x in os.listdir(MODELS_DIRECTORY) if x.endswith(".gguf")],key=os.path.basename)
    models_dicts = []
    for model in models:
        name = model.lower()
        if "gemma" in name: family = "gemma"
        elif "phi" in name: family = "phi"
        elif "llama" in name: family = "llama"
        elif "mistral" in name: family = "mistral"
        elif "glm" in name: family = "glm"
        elif any(k in name for k in ["qwen", "lfm", "bonsai", "jan", "falcon", "diffucoder", "tars", "wedlm", "ggml", "gpt"]):
            family = "chatml"
        else: family = None
        m_dict = {"name": name, "path": f"models/{model}.gguf", "family": family, "params": 122}
        models_dicts.append(m_dict)

    with open(MODELS_FILE, "w") as f:
        json.dump(models_dicts, f, indent=1)

