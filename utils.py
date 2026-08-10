import json
import os
import subprocess
import sys
import re
import tempfile
from contextlib import contextmanager
from llama_cpp.llama_chat_format import Jinja2ChatFormatter
from chat_templates import TEMPLATES_INFERENCE, QWEN_WARMUP, EOS_TOKENS

DEVICES_FILE = "devices.json"
MODELS_FILE = "models/models.json"
MODELS_DIRECTORY = "models"

class MyException(Exception):
    def __init__(self, error_type, message):
        super().__init__(message)
        self.message = message
        self.error_type = error_type

    def __str__(self):
        return f"<ERROR: {self.error_type}> {self.message}"

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
    models = sorted([os.path.basename(x) for x in os.listdir(MODELS_DIRECTORY) if x.endswith(".gguf")],
                    key=os.path.basename)
    current = set([f"models/{x}" for x in models])
    with open(MODELS_FILE, "r") as f:
        models =  json.load(f)
    usable = []
    for model in models:
        if model["path"] in current: usable.append(model)
    return usable


def get_handlers(family: str, custom: bool):
    if not family or not custom: return None, None

    handler_inference = Jinja2ChatFormatter(
        template=TEMPLATES_INFERENCE[family],
        eos_token=EOS_TOKENS[family],
        bos_token=""
    ).to_chat_handler()

    # if family == "chatml":
    #     handler_warmup = Jinja2ChatFormatter(
    #         template=QWEN_WARMUP,
    #         eos_token=EOS_TOKENS[family],
    #         bos_token=""
    #     ).to_chat_handler()
    #
    #     return handler_inference, handler_warmup
    return handler_inference, None

if __name__ == '__main__':
    models = sorted([os.path.basename(x) for x in os.listdir(MODELS_DIRECTORY) if x.endswith(".gguf")],key=os.path.basename)
    print(models)
    reals = set([f"models/{x}" for x in models])
    print(reals)
    # models_dicts = []
    # for model in models:
    #     name = model.lower()
    #     if "gemma" in name: family = "gemma"
    #     elif "phi" in name: family = "phi"
    #     elif "llama" in name: family = "llama"
    #     elif "mistral" in name: family = "mistral"
    #     elif "glm" in name: family = "glm"
    #     elif any(k in name for k in ["qwen", "lfm", "bonsai", "jan", "falcon", "diffucoder", "tars", "wedlm", "ggml", "gpt"]):
    #         family = "chatml"
    #     else: family = None
    #     m_dict = {"name": model.replace("-", " "), "path": f"models/{model}.gguf", "family": family, "params": 122}
    #     models_dicts.append(m_dict)
    #
    # with open(MODELS_FILE, "w") as f:
    #     json.dump(models_dicts, f, indent=1)

    # with open("models/backup.json", "r") as f:
    #     models_dicts = json.load(f)
    # big_dict = {}
    # print(models_dicts)
    # for model in models_dicts:
    #     big_dict[model["path"].split("/")[-1]] = model
    # print(big_dict)
    # with open(MODELS_FILE, "w") as f:
    #     json.dump(models_dicts, f, indent=1)
    #     # print("skibidi")
