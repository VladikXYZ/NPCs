import os
import json
import sys
import time
import pandas
import bench
import platform
from tqdm import tqdm
import questionary
from llama_cpp import Llama
from contextlib import contextmanager

MODEL_DIR = 'models/'
DEVICES_FILE = "devices.json"
CONTEXT_SIZE = 4096
RESPONSE_LENGTH = 32
WARMUP_COUNT = 4

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


class Wrapper:
    def __init__(self, dev=-1):
        self.models = sorted(
            [os.path.splitext(os.path.basename(x))[0] for x in os.listdir(MODEL_DIR) if x.endswith(".gguf")],
            key=os.path.basename)

        if os.path.exists(DEVICES_FILE):
            with open(DEVICES_FILE, "r") as f:
                self.devices = json.load(f)
        else: self.devices = bench.get_devices()

        self.action = None
        self.device = None
        self.selected_model = None
        self.gpu_layers = 0

        self._run_setup_menu(dev)

    def _run_setup_menu(self, dev):
        if dev != -1 and 0 <= dev < len(self.devices):
            self.action = "Benchmark"
            self.device = self.devices[dev]
            print(f"\n⚡ Fast Start: Running Benchmark on {self.device['name']}...")
            self.gpu_layers = -1 if self.device["type"] == "Vulkan" else 0
            os.environ["GGML_VK_VISIBLE_DEVICES"] = str(self.device["id"] * (self.device["type"] == "Vulkan"))
            self._execute_action()
            return

        self.action = questionary.select("Select operation mode:", choices=["Benchmark", "Auto run", "Chat", "Exit"], qmark="⚙️").ask()

        if not self.action or self.action == "Exit":
            print("Exiting...")
            sys.exit(0)

        device_choices = [f"[{i}] {d['type']:<8} | {d['name']}" for i, d in enumerate(self.devices)]

        dev_choice = questionary.select("Select device:", choices=device_choices, qmark="🎮").ask()

        if not dev_choice: sys.exit(0)

        dev_idx = int(dev_choice.split("]")[0][1:])
        self.device = self.devices[dev_idx]

        self.gpu_layers = -1 if self.device["type"] == "Vulkan" else 0
        os.environ["GGML_VK_VISIBLE_DEVICES"] = str(self.device["id"] * (self.device["type"] == "Vulkan"))

        if self.action == "Chat":
            self.selected_model = questionary.select("Select a model for Chat:", choices=self.models, qmark="🤖").ask()

            if not self.selected_model: sys.exit(0)
        else:
            print(f"\n🚀 Mode set to {self.action}. Will iterate through all {len(self.models)} models.")

        self._execute_action()

    def _execute_action(self):
        if self.action == "Benchmark":
            self.benchmark()
        elif self.action == "Auto run":
            # print("\nExecuting Auto run sequence...")
            self.test()
        elif self.action == "Chat":
            print(f"\nStarting Chat with {self.selected_model}...")
            # self.run_chat()

    def load_llm(self, model_path, role):
        print(f"Loading {os.path.basename(model_path)} | ", end="", flush=True)
        try:
            with Silencer():
                llm = Llama(model_path="models/" + model_path + ".gguf", n_gpu_layers=self.gpu_layers, n_ctx=CONTEXT_SIZE, verbose=False)
                # llm = Llama(model_path="models/" + model_path + ".gguf", n_gpu_layers=self.gpu_layers,
                #             n_ctx=CONTEXT_SIZE, verbose=False, type_k=8, type_v=8, flash_attn=True)
                llm.create_chat_completion(role, max_tokens=1)
        except Exception as e:
            print(f"\n{e}\nProbably not enough memory!!")
            return None

        print(f"Loaded! | ", end="", flush=True)
        return llm

    def benchmark(self):
        my_pc_name = platform.node()
        dev_name = self.device["type"] + "_" + "_".join(self.device["name"].split()[:4])
        print(dev_name)

        LOG_DIR = f'vlad/bench_logs/{my_pc_name}/'
        os.makedirs(LOG_DIR, exist_ok=True)
        log = []

        with open("vlad/test.json", "r") as f:
            messages = json.load(f)

        with open("data_3npcs.json") as file:
            npc = json.load(file)[2]

        chat_history = [{"role": "system", "content": npc["role"] + npc["shared_system_prompt"]}]
        warmup = [{"role": "system", "content": npc["role"] + npc["shared_system_prompt"]},
                  {"role": "user", "content": "warmup"}]

        num_mess = len(messages)
        num_models = len(self.models)

        test_start = time.perf_counter()

        for i, model in enumerate(self.models):
            llm = self.load_llm(model, warmup)
            ttfts = 0
            tpss = 0
            if llm:
                # real warmup:
                # print(llm.tokenize(chat_history[0]["content"].encode('utf-8')))
                prev_n = len(llm.tokenize(chat_history[0]["content"].encode('utf-8')))
                for _ in range(WARMUP_COUNT): llm.create_chat_completion(warmup, max_tokens=1)
                print("Warmuped !!")
                # print(llm.tokenize(chat_history[0]["content"].encode('utf-8')))

                for user_input in tqdm(messages, desc=f"Testing {i + 1}/{num_models} {model}", unit="prompt"):
                    chat_history.append({"role": "user", "content": user_input})

                    start_time = time.perf_counter()
                    ttft = None
                    t_out = 0

                    stream = llm.create_chat_completion(messages=chat_history, stream=True, max_tokens=RESPONSE_LENGTH)
                    assistant_response = []
                    for chunk in stream:
                        delta = chunk['choices'][0]["delta"]
                        if 'content' in delta:
                            if ttft is None:
                                ttft = time.perf_counter() - start_time
                            assistant_response.append(delta['content'])
                            t_out += 1
                    assistant_response = "".join(assistant_response)

                    total_time = time.perf_counter() - start_time
                    gen_time = total_time - (ttft if ttft else 0)
                    tps = t_out / gen_time if gen_time > 0 else 0

                    ttft = ttft if ttft is not None else -1
                    all_tokens = llm.n_tokens
                    t_in = all_tokens - prev_n - t_out
                    chat_history.append({"role": "assistant", "content": assistant_response})
                    log.append([model, ttft, tps, t_in, t_out, total_time, all_tokens])
                    #, user_input.replace('\n', '|'), assistant_response.replace('\n', '|')
                    prev_n = all_tokens
                    ttfts+=ttft
                    tpss+=tps
                
                print(f"Mean TTFT:{ttfts/num_mess:.3f}, Mean T/s: {tpss/num_mess:.3f}")
                del llm
                chat_history = chat_history[:1]
            else:
                for _ in range(num_mess): log.append([-1, -1, -1, -1, -1, -1, -1])
        

        print(f"It all took: {time.perf_counter() - test_start}")

        xd = pandas.DataFrame(log, columns=["MODEL", "TTFT", "T/s", "USER TOKENS", "NPC TOKENS", "TOTAL TIME",
                                            "ALL TOKENS"])
        xd = xd.round(4)
        file_path = f"{LOG_DIR}{dev_name}.csv"
        print(file_path)
        xd.to_csv(file_path, index=False)

    def test(self):
        with open("vlad/test.json", "r") as f:
            messages = json.load(f)

        with open("data_3npcs.json") as file:
            npc = json.load(file)[2]

        chat_history = [{"role": "system", "content": npc["role"] + npc["shared_system_prompt"]}]
        warmup = [{"role": "system", "content": npc["role"] + npc["shared_system_prompt"]},
                  {"role": "user", "content": "warmup!"}]

        num_mess = len(messages)
        num_models = len(self.models)
        log = []

        for i, model in enumerate(self.models):
            llm = self.load_llm(model, warmup)
            print("Ready!!")
            if llm:
                for user_input in tqdm(messages, desc=f"Testing {i + 1}/{num_models} {model}", unit="prompt"):
                    chat_history.append({"role": "user", "content": user_input})

                    response_dict = llm.create_chat_completion(messages=chat_history, max_tokens=RESPONSE_LENGTH)
                    # print(response_dict)
                    assistant_response = response_dict['choices'][0]['message']
                    # print(assistant_response)
                    chat_history.append(assistant_response)
                    log.append([model ,user_input.replace('\n', '|'), assistant_response["content"].replace('\n', '|')])
                
                del llm
                chat_history = chat_history[:1]

        xd = pandas.DataFrame(log, columns=["MODEL", "USER INPUT", "RESPONSE"])
        file_path = f"vlad/bench_logs/test.csv"
        # print(file_path)
        xd.to_csv(file_path, index=False)


if __name__ == '__main__':
    dev = -1
    if len(sys.argv) == 2: dev = int(sys.argv[1])

    Wrapper(dev)