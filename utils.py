import json
import os
import subprocess
import sys
import re
import tempfile

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

@contextmanager
def Capturer():
    sys.stdout.flush(); sys.stderr.flush()
    fd_out, fd_err = sys.stdout.fileno(), sys.stderr.fileno()
    old_out, old_err = os.dup(fd_out), os.dup(fd_err)
    
    with tempfile.TemporaryFile() as tmp:
        os.dup2(tmp.fileno(), fd_out)
        os.dup2(tmp.fileno(), fd_err)
        
        logs = [""]
        try:
            yield logs
        finally:
            sys.stdout.flush(); sys.stderr.flush()
            os.dup2(old_out, fd_out); os.dup2(old_err, fd_err)
            os.close(old_out); os.close(old_err)
            
            tmp.seek(0)
            logs[0] = tmp.read().decode(errors='replace').strip()
            # lines = [line for line in raw_text.split('\n') if line.strip()]
            # logs[0] = '\n'.join(lines[-2:]) if lines else "No C++ logs captured."

def _first_run():
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
        devices = get_devices()
    return devices

def get_models():
    return sorted([os.path.splitext(os.path.basename(x))[0] for x in os.listdir(MODELS_DIRECTORY) if x.endswith(".gguf")],key=os.path.basename)