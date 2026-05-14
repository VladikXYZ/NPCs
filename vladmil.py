import os
import sys
import json
import subprocess
import re

DEVICES_FILE = "devices.json"
class Models:
    def __init__(self):
        self.models_dir: str = "models/"
        print("🔍 Scanning models...")
        self.model_paths = [x for x in os.listdir(self.models_dir) if x.endswith(".gguf")]
        self.devices = self._get_hardware_options(False)
        self.selected_device = self.select_device()

    def _get_hardware_options(self, reload):
        if os.path.exists(DEVICES_FILE) and not reload:
            print("Loading devices from config file...")
            with open(DEVICES_FILE, "r") as f:
                return json.load(f)
        else:
            print("🔍 Scanning hardware... (this takes a second)")
            print(self.models_dir + self.models[0])
            script = f"""
import sys
from llama_cpp import Llama
try:
   llm = Llama(model_path='models/gemma-4-E2B-it-Q4_K_M.gguf', n_gpu_layers=1, verbose=True)
except Exception:
   pass
"""
            result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                                    encoding='utf-8')
            devices = []
            for line in result.stderr.split('\n'):
                match = re.search(r"ggml_vulkan:\s+(\d+)\s+=\s+(.*?)\s+\|", line)
                if match:
                    print(line)
                    devices.append({"id": match.group(1), "name": match.group(2).strip(), "type": "Vulkan"})

            cpu_id = str(len(devices))
            devices.append({"id": cpu_id, "name": "CPU", "type": "CPU"})

            with open(DEVICES_FILE, "w") as f:
                json.dump(devices, f)
            return devices

    def run_martin(self):
        for model_path in self.model_paths:
            llm = self.load_llm(model_path)
            print(llm.n_tokens)
            break

    def load_llm(self, model_path):
        is_vulkan =  (self.selected_device["type"] == "Vulkan")
        gpu_layers = -1 * is_vulkan
        os.environ["GGML_VK_VISIBLE_DEVICES"] = self.selected_device["id"] * is_vulkan
        from llama_cpp import Llama
        try: return Llama(model_path=self.models_dir+model_path, n_gpu_layers=gpu_layers, n_ctx=4096, verbose=False)
        except Exception as e:
            print(f"❌ [WORKER] Crash / Out of Memory: {e}")
            sys.exit(0)

    def select_device(self, val = -1):
        first_time = True
        idx = val
        while True:
            if 0 <= idx < len(self.devices): return self.devices[idx]
            if first_time:
                first_time = False
                print("\n" + "=" * 60)
                print("🎮 AVAILABLE ACCELERATORS")
                print("=" * 60)
                print(f" {'ID':<3} | {'Type':<8} | {'Device Name'}")
                print("-" * 60)
                for i, d in enumerate(self.devices):
                    print(f" [{i}] | {d['type']:<8} | {d['name']}")
                print("=" * 60)
            idx = self._get_int(input("Select device: "))

    def _get_int(self, val):
        try: return int(val)
        except ValueError: return -1


models = Models()
# models.select_device()
models.run_martin()