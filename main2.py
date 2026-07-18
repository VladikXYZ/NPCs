import os
import json
import sys
import time
import pandas
import bench2
import gc
from tqdm import tqdm
import questionary
from llama_cpp import Llama

MODEL_DIR = 'models/'
DEVICES_FILE = "devices.json"
CONTEXT_SIZE = 4096


class Wrapper:
    def __init__(self, dev=-1):
        self.models = sorted([os.path.splitext(os.path.basename(x))[0] for x in os.listdir(MODEL_DIR) if x.endswith(".gguf")], key=os.path.basename)
        if os.path.exists(DEVICES_FILE):
            with open(DEVICES_FILE, "r") as f:
                self.devices = json.load(f)
        else:
            self.devices = bench2.get_devices()
        self.device = self.select_device(dev)
        self.gpu_layers = -1 if self.device["type"] == "Vulkan" else 0
        os.environ["GGML_VK_VISIBLE_DEVICES"] = self.device["id"] * (self.device["type"] == "Vulkan")

    def _get_int(self, val):
        try:
            return int(val)
        except ValueError:
            return -1

    def select_device(self, val=-1):
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
                for i, d in enumerate(self.devices): print(f" [{i}] | {d['type']:<8} | {d['name']}")
                print("=" * 60)
            idx = self._get_int(input("Select device: "))

    def load_llm_with_warmup(self, model_path, role):
        class Silencer:
            def __init__(self, suppress=True):
                self.suppress = suppress
                self.old_stderr = None
                self.devnull = None

            def __enter__(self):
                if self.suppress:
                    self.old_stderr = os.dup(sys.stderr.fileno())
                    self.devnull = os.open(os.devnull, os.O_WRONLY)
                    os.dup2(self.devnull, sys.stderr.fileno())
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                if self.suppress and self.old_stderr is not None:
                    os.dup2(self.old_stderr, sys.stderr.fileno())
                    os.close(self.old_stderr)
                    os.close(self.devnull)
                return False

        print(f"Loading {os.path.basename(model_path)} | ", end="", flush=True)
        try:
            with Silencer():
                llm = Llama(model_path="models/"+model_path+".gguf", n_gpu_layers=self.gpu_layers, n_ctx=CONTEXT_SIZE, verbose=False)
                llm.create_chat_completion(role, max_tokens=1)
        except Exception as e:
            print(e)
            print(f"Not enough memory!!")

            return None
        print(f"Loaded!")
        return llm


    def run_test(self):
        import platform
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
        warmup = [{"role": "system", "content": npc["role"] + npc["shared_system_prompt"]}, {"role": "user", "content": "warmup!"}]
        # print(warmup)

        num_mess = len(messages)
        num_models = len(self.models)
        print(f"Testing:")
        for model in self.models:
            print(model)
        for i, model in enumerate(self.models):
            # if i != 6: continue
            llm = self.load_llm_with_warmup(model, warmup)
            if llm:
                prev_n = llm.n_tokens
                # model_info = os.path.basename(model)[:-5]
                for user_input in tqdm(messages, desc=f"Testing {i+1}/{num_models} {model}", unit="prompt"):
                    chat_history.append({"role": "user", "content": user_input})

                    start_time = time.perf_counter()
                    first_token_time = None
                    token_count = 0

                    stream = llm.create_chat_completion(messages=chat_history, stream=True, max_tokens=128)
                    assistant_response = ""
                    for chunk in stream:
                        delta = chunk['choices'][0].get('delta', {})
                        if 'content' in delta:
                            if first_token_time is None:
                                first_token_time = time.perf_counter() - start_time
                            assistant_response += delta['content']
                            token_count += 1

                    total_time = time.perf_counter() - start_time
                    gen_time = total_time - (first_token_time if first_token_time else 0)
                    tps = token_count / gen_time if gen_time > 0 else 0

                    chat_history.append({"role": "assistant", "content": assistant_response})
                    first_token_time = first_token_time if first_token_time is not None else 0.0
                    all_tokens = llm.n_tokens
                    log.append(
                        [first_token_time, tps, token_count, all_tokens - prev_n - token_count, total_time, all_tokens])
                    prev_n = all_tokens
                del llm
                chat_history = chat_history[:1]
            else:
                for _ in range(num_mess): log.append([-1, -1, -1, -1, -1, -1])
        xd = pandas.DataFrame(log, columns=["TTFT", "T/S", "NPC TOKENS", "USER TOKENS", "TOTAL TIME", "ALL TOKENS"])
        file_path = f"{LOG_DIR}{dev_name}.csv"
        print(file_path)
        xd.to_csv(file_path, index=False)

if __name__ == '__main__':
    dev = -1
    if len(sys.argv) == 2:
        dev = int(sys.argv[1])
        wrap = Wrapper(dev)
        wrap.run_test()
    wrap = Wrapper(dev)
    # wrap.run_test()
    # wrap.run_rolloj()
    # wrap.run_martin()

