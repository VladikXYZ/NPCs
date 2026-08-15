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
from utils import get_devices, get_models, Silencer, Catcher, get_handlers, MyException

# MODEL_DIR = 'models/'
# DEVICES_FILE = "devices.json"
PC_NAME = platform.node()
LOG_DIR = f'benchmarks/{PC_NAME}/'
LOG_DIR = ""
# os.makedirs(LOG_DIR, exist_ok=True)
with open("vlad/test.json", "r") as f: MESSAGES = json.load(f)
with open("data_3npcs.json") as file: NPC = json.load(file)[2]
CUSTOM_JINJA = True
if CUSTOM_JINJA:
    CHAT_HISTORY = [{"role": "system", "content": NPC["role"]}]
    WARMUP = CHAT_HISTORY[:]
else:
    CHAT_HISTORY = [{"role": "system", "content": NPC["role"] + NPC["shared_system_prompt"]}]
    WARMUP = CHAT_HISTORY[:]
    WARMUP.append({"role": "user", "content": "warmup"})

NUM_MESS = len(MESSAGES)
CONTEXT_SIZE = 4096
MAX_TOKENS = 32
WARMUP_COUNT = 4
TIMEOUT = (NUM_MESS * (0.9 + (MAX_TOKENS / 5.5))).__ceil__()
# TIMEOUT = 4
HEADER = ["MODEL", "TTFT", "T/s", "USER TOKENS", "NPC TOKENS", "TOTAL TIME", "ALL TOKENS", "PROMPT", "RESPONSE"]
ERROR_ROW = [-1 for _ in range(len(HEADER)-2)]


RED = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"


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

    def load_llm(self, model):
        # print(f"Loading {os.path.basename(model["path"])} | ", end="", flush=True)
        print(f"Loading {model["name"]} | ", end="", flush=True)
        llm_kwargs = { "model_path": model["path"], "n_gpu_layers": self.gpu_layers,
                    "n_ctx": CONTEXT_SIZE, "verbose": False, "temperature": 0 }


        infer, warmup = get_handlers(model["family"], CUSTOM_JINJA)
        if warmup: llm_kwargs["chat_handler"] = warmup
        elif infer: llm_kwargs["chat_handler"] = infer
        llm, err = None, None
        with Silencer():
            try:
                
                llm = Llama(**llm_kwargs)
                print(f"Loaded! | ", end="", flush=True)

                try:
                    llm.create_chat_completion(WARMUP, max_tokens=1)
                    if warmup: llm.chat_handler = infer
                    print("Warmuped!!", flush=True)
                
                except Exception as e:
                    if hasattr(llm, 'close'): llm.close()
                    del llm
                    err = f"Crashed during generation: {e}"
            except Exception as e:
                err = f"Crashed during loading: {e}"


        if err:
            llm_kwargs["verbose"] = True
            with Catcher() as c:
                try:
                    llm = Llama(**llm_kwargs)   
                    llm.create_chat_completion(WARMUP, max_tokens=1)
                except: llm = None
            print("")
            raise MyException(err, c[0])
        return llm, err


        

    def _run_benchmark(self):
        dev_name = self.device["type"] + "_" + "_".join(self.device["name"].split())

        log = []
        num_models = len(self.models)
        test_start = time.perf_counter()
        chat_history = CHAT_HISTORY[:]
        print("DEViCE:", dev_name)
        print("TIMEOUT:", TIMEOUT)
        print("MODELS:", num_models)

        for i, model in enumerate(self.models):
            family = model["family"]
            model_log = []
            llm = None
            try:
                llm, err = self.load_llm(model)
                model_start = time.perf_counter()
                prev_n = llm.n_tokens

                for user_input in tqdm(MESSAGES, desc=f"Testing {i + 1}/{num_models} {model["name"]}", unit="prompt"):
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
                        else: raise MyException("Timeout!", f"Ran out of time ({TIMEOUT} s)")
                    assistant_response = "".join(assistant_response)

                    total_time = time.perf_counter() - start_time
                    gen_time = total_time - ttft
                    tps = t_out / gen_time if gen_time > 0 else -1
                    all_tokens = llm.n_tokens
                    t_in = all_tokens - prev_n - t_out

                    query = user_input[:].replace('\n', '|')
                    response = assistant_response[:].replace('\n', '|')
                    model_log.append([model["name"], ttft, tps, t_in, t_out, total_time, all_tokens, query, response])
                    chat_history.append({"role": "assistant", "content": assistant_response})
                    prev_n = all_tokens
                print(f"{GREEN}FINISHED!!!{RESET}")

            except MyException as e:
                print(f"{RED}{e.error_type}{RESET}")
                model_log.append([model["name"]] + ERROR_ROW +[str(e)])
                done = len(model_log)
                for _ in range(NUM_MESS - done): model_log.append([model["name"]] + ERROR_ROW + [-1])

            finally:
                if 'stream' in locals(): del stream
                if hasattr(llm, 'close'): llm.close()
                del llm
                chat_history = chat_history[:1]
                log.extend(model_log)

        print(f"It all took: {time.perf_counter() - test_start}")

        xd = pandas.DataFrame(log, columns=HEADER)
        xd = xd.round(3)
        file_path = f"{LOG_DIR}{dev_name}.csv"
        print(f"Saved to {file_path}")
        xd.to_csv(file_path, index=False, float_format="%.3f")



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
    else: Benchmarker()
