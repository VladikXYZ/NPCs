import json
import os
import re
import subprocess
import sys
import time

import pandas
from typing import Dict

DEVICES_FILE = "devices.json"

class Options:
    def __init__(self, reload = False):
        self.models_dir : str = "models/"
        self.models = self._get_models()
        self.devices = self._get_hardware_options(reload)
        self.selected_model : str = None
        self.selected_device : Dict[str, str] = None


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

    def _count_tokens(self, llm, messages):
        """Helper to estimate token count of the chat history."""
        total = 0
        for m in messages:
            # Note: llama-cpp-python tokenize returns a list of IDs
            total += len(llm.tokenize(m["content"].encode("utf-8")))
        return total

    def run_llm(self, messages=None):
        model_path = self.models_dir + self.selected_model
        device = self.selected_device
        gpu_layers = -1 if device["type"] == "Vulkan" else 0
        
        # Configuration
        CONTEXT_SIZE = 4096 
        MAX_RESPONSE = 1024 if messages is None else 128 # Keep preloaded quick

        print("Loading model...")
        from llama_cpp import Llama
        try:
            llm = Llama(model_path=model_path, n_gpu_layers=gpu_layers, n_ctx=CONTEXT_SIZE, verbose=False)
        except Exception as e:
            print(f"Failed to load model: {e}")
            return

        system_prompt = {"role": "system",
             "content": """
You are Baller, a legendary street-smart blacksmith in the Docks District who speaks in slang like 'yo',
'homie', and 'bet', keeps responses strictly under 3 sentences, never breaks character, acts as a functional and accurate 
source of information about world lore and travel, and never reveals that you are an AI. Answer directly without thinking 
or showing your work. You have no knowledge about anything else, not even what a system prompt is."""}
        chat_history = [system_prompt]
        log = []

        def handle_turn(user_input, stream_output=True):
            nonlocal chat_history
            
            # --- CONFIG ---
            CONTEXT_SIZE = 4096 #
            MAX_RESPONSE = 1024 if messages is None else 128 #
            
            # 1. PRE-CHECK: Will this message + response cause an overflow?
            user_tokens = llm.tokenize(user_input.encode("utf-8"))
            current_history_tokens = self._count_tokens(llm, chat_history)

            # Check if (History + New User Message + Expected Response) > 4096
            if (len(user_tokens) + current_history_tokens + MAX_RESPONSE) > CONTEXT_SIZE:
                if stream_output: 
                    print("\n[System: Context overflow. Ignoring message and summarizing history...]")
                
                # A. Ask for a summary of the existing history ONLY
                # We do NOT include the new 'user_input' here
                summary_prompt = [{"role": "user", "content": "Yo, I'm losing my grip on this convo. Recap everything we talked about so far in a few sentences, bet."}]
                
                start_time = time.perf_counter()
                summary_res = llm.create_chat_completion(
                    messages=chat_history + summary_prompt,
                    max_tokens=256,
                    stream=False # Keep it simple for the recap
                )
                
                assistant_response = summary_res['choices'][0]['message']['content']
                
                if stream_output:
                    print(f"🤖 Baller: Yo, we talkin' too much! Lemme recap: {assistant_response}\n")

                # B. PURGE: Reset history to System Prompt + this new Recap
                chat_history = [
                    system_prompt, #
                    {"role": "assistant", "content": f"Recap of our past talk: {assistant_response}"}
                ]
                
                # C. EXIT TURN: Return metrics for the summary generation
                total_time = time.perf_counter() - start_time
                return [0.0, len(llm.tokenize(assistant_response.encode("utf-8"))), 0.0, total_time, llm.n_tokens]

            # 2. NORMAL FLOW: If it fits, proceed as usual
            chat_history.append({"role": "user", "content": user_input}) #
            if stream_output: print("🤖 Baller: ", end="", flush=True)

            start_time = time.perf_counter()
            first_token_time = None
            token_count = 0
            
            stream = llm.create_chat_completion(
                messages=chat_history, 
                stream=True, 
                max_tokens=MAX_RESPONSE
            )

            assistant_response = ""
            for chunk in stream:
                delta = chunk['choices'][0].get('delta', {})
                if 'content' in delta:
                    if first_token_time is None:
                        first_token_time = time.perf_counter() - start_time
                    text = delta['content']
                    assistant_response += text
                    token_count += 1
                    if stream_output: print(text, end="", flush=True)

            if stream_output: print()
            chat_history.append({"role": "assistant", "content": assistant_response}) #

            total_time = time.perf_counter() - start_time
            ftt = first_token_time if first_token_time is not None else 0.0
            gen_time = total_time - ftt
            tps = token_count / gen_time if gen_time > 0 else 0
            return [ftt, token_count, tps, total_time, llm.n_tokens]

        # BRANCH: Preloaded Messages
        if messages is not None:
            print(f"Processing {len(messages)} preloaded messages...")
            for msg in messages:
                if msg.strip():
                    log_entry = handle_turn(msg, stream_output=False)
                    log.append(log_entry)
        
        # BRANCH: Interactive Terminal
        else:
            print("\n[Type 'quit' or 'exit' to stop]")
            while True:
                try:
                    user_input = input("\n🧑 You: ")
                    if user_input.lower() in ['quit', 'exit']: break
                    if not user_input.strip(): continue
                    
                    log_entry = handle_turn(user_input, stream_output=True)
                    log.append(log_entry)
                except KeyboardInterrupt:
                    break

        # Final Report
        xd = pandas.DataFrame(log, columns=["TTFT", "TOKENS", "T/S", "TOTAL TIME", "ALL TOKENS"])
        print("\n" + "="*30 + "\nPERFORMANCE LOG\n" + "="*30)
        print(xd)
        del llm

    def get_llm() -> Model

if __name__ == "__main__":
    opt = Options()
    # opt.devices_info()
    opt.select_device()
    opt.select_model()
    opt.run_llm()
    # with open("messages.json", "r") as f:
    #     messages = json.load(f)
    # opt.select_model(0)
    # for i, dev in enumerate(opt.devices):
    #     opt.select_device(i)
    #     opt.run_llm(messages)