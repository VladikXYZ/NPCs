import json
import os
import re
import subprocess
import sys
import time

import pandas

DEVICES_FILE = "devices.json"

class Options:
    def __init__(self, reload = False):
        self.models_dir = "models/"
        self.models = self._get_models()
        self.devices = self._get_hardware_options(reload)
        self.selected_model = None
        self.selected_device = None


    def _get_models(self) -> list[str]:
            print("🔍 Scanning models...")
            models = [x for x in os.listdir(self.models_dir) if x.endswith(".gguf")]
            return models

    def _get_hardware_options(self, reload):
        if os.path.exists(DEVICES_FILE) and not reload:
            print("Loading devices from config file...")
            with open(DEVICES_FILE, "r") as f:
                return json.load(f)
        else:
            print("🔍 Scanning hardware... (this takes a second)")
            print(self.models_dir+self.models[0])
            script = f"""
import sys
from llama_cpp import Llama
try:
   llm = Llama(model_path='models/gemma-4-E2B-it-Q4_K_M.gguf', n_gpu_layers=1, verbose=True)
except Exception:
   pass
"""
            result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, encoding='utf-8')
            devices = []
            # print(result.stderr)
            for line in result.stderr.split('\n'):
                match = re.search(r"ggml_vulkan:\s+(\d+)\s+=\s+(.*?)\s+\|", line)
                if match:
                    print(line)
                    devices.append({"id": match.group(1), "name": match.group(2).strip(), "type": "Vulkan"})

            cpu_id = str(len(devices))
            devices.append({ "id": cpu_id, "name": "CPU", "type": "CPU" })

            with open(DEVICES_FILE, "w") as f:
                json.dump(devices, f)
            return devices

    def _get_int(self, val):
        try: return int(val)
        except ValueError: return -1

    def select_model(self, val = -1):
        first_time = True
        done = False
        idx = self._get_int(val)
        if 0 <= idx < len(self.models):
            done = True
            self.selected_model = self.models[idx]
            print("First time selected")
        while not done:
            if first_time:
                first_time = False
                self.models_info()

            idx = self._get_int(input("Select model: "))
            if 0 <= idx < len(self.models):
                done = True
                self.selected_model = self.models[idx]
                print("Selected from terminal")

    def select_device(self, val = -1):
        first_time = True
        done = False
        idx = self._get_int(val)
        if 0 <= idx < len(self.devices):
            done = True
            self.selected_device = self.devices[idx]
            print("First time selected")
        while not done:
            if first_time:
                first_time = False
                self.devices_info()

            idx = self._get_int(input("Select device: "))
            if 0 <= idx < len(self.devices):
                done = True
                self.selected_device = self.devices[idx]
                print("Selected from terminal")


    def model(self): return self.models_dir + self.selected_model
    def device(self): return self.selected_device

    def run_llm(self, messages = [""]):

        model = self.models_dir+self.selected_model
        device = self.selected_device

        gpu_layers= -1*(device["type"] == "Vulkan")
        main_gpu = int(device["id"])*(device["type"] == "Vulkan")
        dev_id = int(device["id"])
        if dev_id == 2:
            print("Setting os.environ")
            os.environ["GGML_VK_VISIBLE_DEVICES"] = device["id"]
            main_gpu = 0
        print(gpu_layers, main_gpu)

        print("Loading model... (this might take a moment)")
        from llama_cpp import Llama, llama_cpp
        try:
            llm = Llama(model_path=model, n_gpu_layers=gpu_layers, n_ctx=4096, verbose=False)
        except Exception as e:
            print(f"Failed to load model: {e}")
            print("Not enough memory")
            return
        print("Loaded!!!")
        chat_history = [
            {"role": "system",
             "content": "You are Baller, a legendary street-smart blacksmith in the Docks District who speaks in slang like 'yo', 'homie', and 'bet', keeps responses strictly under 3 sentences, never breaks character, acts as a functional and accurate source of information about world lore and travel, and never reveals that you are an AI. Answer directly without thinking or showing your work."}]

        log = []
        for user_input in messages:
            if not user_input.strip():
                continue
            chat_history.append({"role": "user", "content": user_input})
            start_time = time.perf_counter()
            first_token_time = None
            token_count = 0
            stream = llm.create_chat_completion(messages=chat_history, stream=True, max_tokens=128)

            assistant_response = ""
            for chunk in stream:
                delta = chunk['choices'][0].get('delta', {})
                if 'content' in delta:
                    if first_token_time is None: first_token_time = time.perf_counter() - start_time
                    text = delta['content']
                    assistant_response += text
                    token_count += 1

            total_time = time.perf_counter() - start_time
            gen_time = total_time - (first_token_time if first_token_time else 0)
            tps = token_count / gen_time if gen_time > 0 else 0
            chat_history.append({"role": "assistant", "content": assistant_response})
            first_token_time = first_token_time if first_token_time is not None else 0.0
            log.append([first_token_time, token_count, tps, total_time, llm.n_tokens])

        xd = pandas.DataFrame(log, columns=["TTFT", "TOKENS", "T/S", "TOTAL TIME", "ALL TOKENS"], index=None)
        print(xd)
        del llm

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
        print("=" * 60)

if __name__ == "__main__":
    opt = Options()
    opt.devices_info()
    with open("messages.json", "r") as f:
        messages = json.load(f)
    opt.select_model(0)
    for i, dev in enumerate(opt.devices):
        opt.select_device(i)
        opt.run_llm(messages)