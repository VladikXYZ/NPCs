import subprocess
import os
import json
import sys
import time
import pandas
import platform
import questionary
from tqdm import tqdm
from llama_cpp import Llama
from utils import get_devices, get_models, Silencer

MODEL_DIR = 'models/'
DEVICES_FILE = "devices.json"
PC_NAME = platform.node()
LOG_DIR = f'vlad/bench_logs/{PC_NAME}/'
os.makedirs(LOG_DIR, exist_ok=True)
with open("vlad/test.json", "r") as f: MESSAGES = json.load(f)
with open("data_3npcs.json") as file: NPC = json.load(file)[2]

CHAT_HISTORY = [{"role": "system", "content": NPC["role"] + NPC["shared_system_prompt"]}]
WARMUP = CHAT_HISTORY[:]
WARMUP.append({"role": "user", "content": "warmup"})
NUM_MESS = len(MESSAGES)
CONTEXT_SIZE = 4096
MAX_TOKENS = 128
WARMUP_COUNT = 4
TIMEOUT = (NUM_MESS * (1 + (MAX_TOKENS / 5))).__ceil__()
HEADER = ["MODEL", "TTFT", "T/s", "USER TOKENS", "NPC TOKENS", "TOTAL TIME", "ALL TOKENS", "PROMPT", "RESPONSE"]
ERROR_ROW = [-1 for _ in range(len(HEADER)-1)]


class Benchmarker:
    def __init__(self, dev=None):
        self.models = get_models()
        self.devices = get_devices()

        if dev is None:
            device_choices = [f"{i} | {d['type']:<8} | {d['name']}" for i, d in enumerate(self.devices)]
            dev_choice = questionary.select("Select device:", choices=device_choices, qmark="🎮").ask()
            if not dev_choice: sys.exit("Exiting...")
            dev_idx = int(dev_choice.split("|")[0])
            self.device = self.devices[dev_idx]
        else:
            if 0 <= dev < len(self.devices):self.device = self.devices[dev]
            else: sys.exit(f"Invalid device value:{dev}")

        print(f"\n⚡ Fast Start: Running Benchmark on {self.device['name']}...")
        self.gpu_layers = -1 if self.device["type"] == "Vulkan" else 0
        os.environ["GGML_VK_VISIBLE_DEVICES"] = str(self.device["id"] * (self.device["type"] == "Vulkan"))
        self._run_benchmark()




    def load_llm(self, model_path):
        print(f"Loading {os.path.basename(model_path)} | ", end="", flush=True)
        try:
            with Silencer():
                llm = Llama(model_path="models/" + model_path + ".gguf", n_gpu_layers=self.gpu_layers, n_ctx=CONTEXT_SIZE, verbose=False)
                # llm = Llama(model_path="models/" + model_path + ".gguf", n_gpu_layers=self.gpu_layers,
                #             n_ctx=CONTEXT_SIZE, verbose=False, type_k=8, type_v=8, flash_attn=True)
                llm.create_chat_completion(WARMUP, max_tokens=1)
        except Exception as e:
            print(f"\n{e}\nProbably not enough memory!!")
            return None

        print(f"Loaded! | ", end="", flush=True)
        return llm

    def _run_benchmark(self):
        dev_name = self.device["type"] + "_" + "_".join(self.device["name"].split())
        print("DEViCE:",dev_name)
        log = []
        num_models = len(self.models)
        test_start = time.perf_counter()
        print("TIMEOUT:", TIMEOUT)
        chat_history = CHAT_HISTORY[:]
        for i, model in enumerate(self.models):
            row = [model] + ERROR_ROW
            failed = False
            model_log = []
            llm = self.load_llm(model)
            if llm:
                try:
                    prev_n = len(llm.tokenize(chat_history[0]["content"].encode('utf-8')))
                    for _ in range(WARMUP_COUNT): llm.create_chat_completion(WARMUP, max_tokens=1)
                    print("Warmuped!!")
                    model_start = time.perf_counter()

                    for user_input in tqdm(MESSAGES, desc=f"Testing {i + 1}/{num_models} {model}", unit="prompt"):

                        chat_history.append({"role": "user", "content": user_input})
                        ttft, t_out = TIMEOUT*2, 0
                        start_time = time.perf_counter()
                        assistant_response = [""] * MAX_TOKENS

                        stream = llm.create_chat_completion(messages=chat_history, stream=True, max_tokens=MAX_TOKENS)

                        for chunk in stream:
                            current = time.perf_counter()
                            if current - model_start <= TIMEOUT:
                                delta = chunk['choices'][0]["delta"]
                                if 'content' in delta:
                                    ttft = min(current-start_time, ttft)
                                    assistant_response[t_out] = delta['content']
                                    t_out += 1
                            else:
                                failed = True
                                done = len(model_log)
                                for _ in range(NUM_MESS - done): model_log.append(row)
                                break

                        if failed: break
                        assistant_response = "".join(assistant_response)
                        chat_history.append({"role": "assistant", "content": assistant_response})

                        total_time = time.perf_counter() - start_time
                        gen_time = total_time - ttft
                        tps = t_out / gen_time if gen_time > 0 else -1
                        all_tokens = llm.n_tokens
                        t_in = all_tokens - prev_n - t_out
                        prev_n = all_tokens

                        model_log.append([model, ttft, tps, t_in, t_out, total_time, all_tokens,
                                          user_input.replace('\n', '|'), assistant_response.replace('\n', '|')])

                except Exception as e:
                    print(f"\n{e}\nFailed due error.")

                if failed:
                    print(f"❌ {model} failed due to timeout.")
                # else:
                #     print(f"Mean TTFT:{ttfts / num_mess:.3f}, Mean T/s: {tpss / num_mess:.3f}")
                del llm
                chat_history = chat_history[:1]
                log.extend(model_log)
            else:
                log.extend([row for _ in range(NUM_MESS)])

        print(f"It all took: {time.perf_counter() - test_start}")

        xd = pandas.DataFrame(log, columns=HEADER)
        xd = xd.round(3)
        file_path = f"{LOG_DIR}{dev_name}.csv"
        print(f"Saved to {file_path}")
        xd.to_csv(file_path, index=False)



if __name__ == '__main__':
    if len(sys.argv) == 2:
        num = int(sys.argv[1])
        if num == -1:
            devices = get_devices()
            start = time.time()
            for j in range(len(devices)):
                prev = time.time()
                subprocess.run([sys.executable, "bench.py", str(j)])
                print(f"This took {time.time() - prev} seconds")
            print(f"All tests took {time.time() - start} seconds")
        else: Benchmarker(num)
    else:
        Benchmarker()
