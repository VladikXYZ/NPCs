import os
import json
import sys
import time
import pandas
import bench
from tqdm import tqdm
import questionary
from llama_cpp import Llama
from contextlib import contextmanager

MODEL_DIR = 'models/'
DEVICES_FILE = "devices.json"
CONTEXT_SIZE = 4096


class Wrapper:
    def __init__(self, dev=-1):
        # 1. Load models and sort them by basename
        self.models = sorted(
            [os.path.splitext(os.path.basename(x))[0] for x in os.listdir(MODEL_DIR) if x.endswith(".gguf")],
            key=os.path.basename)

        # 2. Load devices from JSON or fallback to bench2
        if os.path.exists(DEVICES_FILE):
            with open(DEVICES_FILE, "r") as f:
                self.devices = json.load(f)
        else:
            self.devices = bench.get_devices()

        # State variables
        self.action = None
        self.device = None
        self.selected_model = None
        self.gpu_layers = 0

        # 3. Run the setup and automatically execute
        self._run_setup_menu(dev)

    def _run_setup_menu(self, dev):
        """Handles the interactive selection, or skips it if a CLI argument is provided."""

        # 🚀 FAST PATH: If you passed a device ID via command line (e.g., python main.py 0)
        if dev != -1 and 0 <= dev < len(self.devices):
            self.action = "Benchmark"
            self.device = self.devices[dev]
            print(f"\n⚡ Fast Start: Running Benchmark on {self.device['name']}...")
            self.gpu_layers = -1 if self.device["type"] == "Vulkan" else 0
            os.environ["GGML_VK_VISIBLE_DEVICES"] = str(self.device["id"] * (self.device["type"] == "Vulkan"))
            self._execute_action()
            return

        # 🐢 INTERACTIVE PATH: If no argument was passed, show the menus
        # Step A: Select the mode
        self.action = questionary.select(
            "Select operation mode:",
            choices=["Benchmark", "Auto run", "Chat", "Exit"],
            qmark="⚙️"
        ).ask()

        if not self.action or self.action == "Exit":
            print("Exiting...")
            sys.exit(0)

        # Step B: Select the device
        device_choices = [
            f"[{i}] {d['type']:<8} | {d['name']}" for i, d in enumerate(self.devices)
        ]
        print()  # For clean line spacing
        dev_choice = questionary.select(
            "Select device:",
            choices=device_choices,
            qmark="🎮"
        ).ask()

        if not dev_choice:
            sys.exit(0)

        dev_idx = int(dev_choice.split("]")[0][1:])
        self.device = self.devices[dev_idx]

        # Apply device hardware configurations
        self.gpu_layers = -1 if self.device["type"] == "Vulkan" else 0
        os.environ["GGML_VK_VISIBLE_DEVICES"] = str(self.device["id"] * (self.device["type"] == "Vulkan"))

        # Step C: Select model (Conditioned on Chat Mode)
        if self.action == "Chat":
            print()  # For clean line spacing
            self.selected_model = questionary.select(
                "Select a model for Chat:",
                choices=self.models,
                qmark="🤖"
            ).ask()

            if not self.selected_model:
                sys.exit(0)
        else:
            print(f"\n🚀 Mode set to {self.action}. Will iterate through all {len(self.models)} models.")

        # Step D: Automatically execute the selected action
        self._execute_action()

    def _execute_action(self):
        """Routes to the corresponding method based on the selected action."""
        if self.action == "Benchmark":
            self.run_test()
        elif self.action == "Auto run":
            print("\nExecuting Auto run sequence...")
            # self.run_auto()
        elif self.action == "Chat":
            print(f"\nStarting Chat with {self.selected_model}...")
            # self.run_chat()

    def load_llm(self, model_path, role):
        @contextmanager
        def Silencer(suppress=True):
            if suppress:
                old_stderr = os.dup(sys.stderr.fileno())
                devnull = os.open(os.devnull, os.O_WRONLY)
                os.dup2(devnull, sys.stderr.fileno())
                try:
                    yield
                finally:
                    # This always runs, even if your code crashes inside the block
                    os.dup2(old_stderr, sys.stderr.fileno())
                    os.close(old_stderr)
                    os.close(devnull)
            else:
                yield

        print(f"Loading {os.path.basename(model_path)} | ", end="", flush=True)
        try:
            with Silencer():
                llm = Llama(model_path="models/" + model_path + ".gguf", n_gpu_layers=self.gpu_layers,
                            n_ctx=CONTEXT_SIZE, verbose=False)
                llm.create_chat_completion(role, max_tokens=1)
        except Exception as e:
            print(e)
            print(f"Probably not enough memory!!")
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
        warmup = [{"role": "system", "content": npc["role"] + npc["shared_system_prompt"]},
                  {"role": "user", "content": "warmup!"}]

        num_mess = len(messages)
        num_models = len(self.models)

        for i, model in enumerate(self.models):
            llm = self.load_llm(model, warmup)
            if llm:
                prev_n = llm.n_tokens
                for user_input in tqdm(messages, desc=f"Testing {i + 1}/{num_models} {model}",
                                       unit="prompt"):
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
                        [first_token_time, tps, token_count, all_tokens - prev_n - token_count, total_time,
                         all_tokens])
                    prev_n = all_tokens

                del llm
                chat_history = chat_history[:1]
            else:
                for _ in range(num_mess): log.append([-1, -1, -1, -1, -1, -1])

        xd = pandas.DataFrame(log, columns=["TTFT", "T/S", "NPC TOKENS", "USER TOKENS", "TOTAL TIME",
                                            "ALL TOKENS"])
        file_path = f"{LOG_DIR}{dev_name}.csv"
        print(file_path)
        xd.to_csv(file_path, index=False)


if __name__ == '__main__':
    dev = -1
    if len(sys.argv) == 2: dev = int(sys.argv[1])

    Wrapper(dev)