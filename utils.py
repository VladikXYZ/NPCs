import json
import os
import subprocess
import sys
import re

DEVICES_FILE = "devices.json"
MODELS_DIRECTORY = "models"

from contextlib import contextmanager
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

def _first_run():
    print("🔍 Scanning hardware... (this takes a second)")
    script = f"""
import sys
from llama_cpp import Llama
try:
   llm = Llama(model_path='models/Supra-Router-51M-Q4_K_M.gguf', n_gpu_layers=1, verbose=True)
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
        devices = get_devices()
    return devices

def get_models():
    return sorted([os.path.splitext(os.path.basename(x))[0] for x in os.listdir(MODELS_DIRECTORY) if x.endswith(".gguf")],key=os.path.basename)