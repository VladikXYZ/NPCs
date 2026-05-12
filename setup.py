import json
import os
import re
import subprocess
import sys

DEVICES_FILE = "devices.json"
MODELS_FILE = "models.json"

class Options:
    def __init__(self):
        self.models_dir = "models/"
        self.models = self.get_models()
        self.devices = self.get_hardware_options()


    def get_models(self) -> list[str]:
        if os.path.exists(MODELS_FILE):
            print("Loading models from config file...")
            with open(MODELS_FILE, "r") as f:
                return json.load(f)
        else:
            print("🔍 Scanning models...")
            models = [x for x in os.listdir(self.models_dir)]
            with open(MODELS_FILE, "w") as f:
                json.dump(models, f)
            return models

    def get_hardware_options(self):
        if os.path.exists(DEVICES_FILE):
            print("Loading devices from config file...")
            with open(DEVICES_FILE, "r") as f:
                return json.load(f)
        else:
            print("🔍 Scanning hardware... (this takes a second)")
            script = f"""
                import sys
                from llama_cpp import Llama
                try:
                    llm = Llama(model_path='{self.models[0]}', n_gpu_layers=1, verbose=True)
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
            devices.append({ "id": cpu_id, "name": "CPU", "type": "CPU" })

            with open(DEVICES_FILE, "w") as f:
                json.dump(devices, f)
            return devices
    def model(self, idx): return self.models_dir + self.models[idx]
    def device(self, idx): return self.devices[idx]

    def models_info(self):
        print("\n" + "=" * 60)
        print("📦 AVAILABLE MODELS")
        for i, m in enumerate(self.models):
            print(f" [{i}] | {m}")
        print("=" * 60)

    def devices_info(self):
        print("\n" + "=" * 60)
        print("🎮 AVAILABLE ACCELERATORS")
        print("=" * 60)
        print(f" {'ID':<3} | {'Type':<8} | {'Device Name'}")
        print("-" * 60)
        for i, d in enumerate(self.devices):
            print(f" [{i}] | {d['type']:<8} | {d['name']}")
        print("=" * 60 + "\n")