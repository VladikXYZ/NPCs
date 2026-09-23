import json
import os
import re
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from math import inf
from pathlib import Path

from llama_cpp import Llama
import llama_cpp.llama_chat_format as llama_chat_format
from chat_templates import (
    BOS_TOKENS,
    EOS_TOKENS,
    INFERENCE_TYPES,
    NATIVE_TEMPLATE_FAMILY,
    reasoning_mode,
)

ROOT = Path(__file__).resolve().parent
DEVICES_FILE = ROOT / "devices.json"
MODELS_FILE = ROOT / "models/models.json"
MODELS_DIRECTORY = ROOT / "models"

_HF_GENERATION_ANNOTATION = re.compile(r"\{%[-+]?\s*(?:end)?generation\s*[-+]?%\}")
_BaseJinja2ChatFormatter = llama_chat_format.Jinja2ChatFormatter


def _strip_hf_generation_annotations(template: str) -> str:
    """Remove HF loss-mask annotations; they do not affect inference prompts."""
    return _HF_GENERATION_ANNOTATION.sub("", template)


class _GenerationCompatibleJinja2ChatFormatter(_BaseJinja2ChatFormatter):
    """Allow llama-cpp-python's Jinja parser to load HF generation-tagged templates."""

    def __init__(self, template: str, *args, **kwargs):
        super().__init__(_strip_hf_generation_annotations(template), *args, **kwargs)


# llama-cpp-python eagerly compiles every chat template embedded in GGUF,
# including when the caller supplies our own handler. Its Jinja environment
# does not recognize HF's `{% generation %}` loss-mask annotations. Normalize
# those no-op inference markers in that metadata path while preserving the
# enclosed prompt text.
llama_chat_format.Jinja2ChatFormatter = _GenerationCompatibleJinja2ChatFormatter


class MyException(Exception):
    def __init__(self, error_type, message):
        super().__init__(message)
        self.message = message
        self.error_type = error_type

    def __str__(self):
        return f"<ERROR: {self.error_type.replace("\n", "")}> {self.message.replace("\n", "")}"

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
    result = subprocess.run([sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True, encoding='utf-8')
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

def get_models(include_hidden=False):
    """Return installed, visible models, independently of the working directory."""
    with open(MODELS_FILE, "r", encoding="utf-8") as f:
        models = json.load(f)
    usable = []
    for model in models:
        if model.get("hidden", False) and not include_hidden:
            continue
        path = ROOT / model["path"]
        if path.is_file():
            usable.append({**model, "path": str(path.resolve())})
    return sorted(usable, key=lambda model: (model["name"].casefold(), model["path"]))


def get_handlers(family: str | None, custom: bool, reason=False, *, reasoning=None):
    mode = reasoning_mode(reason if reasoning is None else reasoning)
    if not family or family == NATIVE_TEMPLATE_FAMILY or not custom:
        return None, None
    infer = INFERENCE_TYPES[mode == "mini"]
    if family not in infer:
        raise ValueError(f"Unknown chat template family: {family}")

    handler_inference = _GenerationCompatibleJinja2ChatFormatter(
        template=infer[family],
        eos_token=EOS_TOKENS[family],
        bos_token=BOS_TOKENS.get(family, "")
    ).to_chat_handler()

    # Warm-up must use the exact same serializer and assistant generation prefix.
    return handler_inference, None


def load_llm(model, llm_kwargs, warmup_inputs=None, custom_jinja=False,
             reason=False, log=False, *, reasoning=None):
    mode = reasoning_mode(reason if reasoning is None else reasoning)
    infer, _ = get_handlers(model.get("family"), custom_jinja, reasoning=mode)
    kwargs = dict(llm_kwargs)
    # Sampling belongs to create_chat_completion, not the Llama constructor.
    kwargs.pop("temperature", None)
    if infer:
        kwargs["chat_handler"] = infer
    warmup_inputs = list(warmup_inputs or [{"role": "user", "content": "Hello."}])
    if warmup_inputs[-1]["role"] != "user":
        warmup_inputs.append({"role": "user", "content": "Hello."})
    print(f"Loading {model['name']} | ", end="", flush=True)
    llm = None
    try:
        with Silencer():
            llm = Llama(**kwargs)
            llm.create_chat_completion(
                messages=warmup_inputs, max_tokens=1, temperature=0.0, seed=42
            )
        # Retain warm hardware, discard the warm-up's token history.
        llm.reset()
        print("Loaded and warmed up.", flush=True)
        return llm
    except Exception as exc:
        if llm is not None:
            llm.close()
        # Do not load the model again to capture an error; that can leak memory
        # and the retry could produce a different failure.
        detail = str(exc) if log else "Model load or warm-up failed."
        raise MyException("Load/warmup error", detail) from exc



if __name__ == '__main__':
    pass
    # models = sorted([os.path.basename(x) for x in os.listdir(MODELS_DIRECTORY) if x.endswith(".gguf")],key=os.path.basename)
    # print(models)
    # reals = set([f"models/{x}" for x in models])
    # print(reals)
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
