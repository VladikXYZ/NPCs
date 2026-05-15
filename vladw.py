import os
import json
import sys
import time

import pandas
from tqdm import tqdm
from llama_cpp import Llama
from contextlib import contextmanager

import ctypes
from llama_cpp import Llama, llama_log_set

MODEL_DIR = 'models/'
DEVICES_FILE = "devices.json"
CONTEXT_SIZE = 4096



@ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)
def mute_llama_log(level, message, user_data):
    pass

llama_log_set(mute_llama_log, ctypes.c_void_p())

@contextmanager
def suppress_cpp_warnings():
    # Save the original terminal error stream
    old_stderr = os.dup(sys.stderr.fileno())
    # Open a black hole (devnull)
    devnull = os.open(os.devnull, os.O_WRONLY)
    # Reroute the error stream to the black hole
    os.dup2(devnull, sys.stderr.fileno())
    try:
        yield
    finally:
        # Restore the terminal error stream when done!
        os.dup2(old_stderr, sys.stderr.fileno())
        os.close(old_stderr)
        os.close(devnull)

class Wrapper:
    def __init__(self, dev =-1):
        self.models = [os.path.join(MODEL_DIR,x) for x in os.listdir(MODEL_DIR) if x.endswith(".gguf")]
        with open(DEVICES_FILE, "r") as f:
            self.devices = json.load(f)
        self.device = self.select_device(dev)
        self.gpu_layers = -1 if self.device["type"] == "Vulkan" else 0
        os.environ["GGML_VK_VISIBLE_DEVICES"] = self.device["id"] * (self.device["type"] == "Vulkan")

    def _get_int(self, val):
        try: return int(val)
        except ValueError: return -1

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
        print(f"Loading {os.path.basename(model_path)} | ", end="")
        # print(role)
        try:
            with suppress_cpp_warnings():
                llm = Llama(model_path=model_path, n_gpu_layers=self.gpu_layers, n_ctx=CONTEXT_SIZE, verbose=False)
                llm.create_chat_completion(role,max_tokens=1)
        except Exception as e:
            print(f"Not enough memory!!")
            return None
        print(f"Loaded!")
        return llm
    
    def load_llm_WITHOUT_warmup(self, model_path):
        print(f"Loading {os.path.basename(model_path)}...")
        # print(role)
        try:
            llm = Llama(model_path=model_path, n_gpu_layers=self.gpu_layers, n_ctx=CONTEXT_SIZE, verbose=False)
        except Exception as e:
            print(f"Failed to load model: {e}")
            return None
        print(f"Loaded!")
        return llm

    def run_test(self):
        import platform

        my_pc_name = platform.node()
        # print(f"My PC is called: {my_pc_name}")
        # print(self.device)
        # return
        dev_name =self.device["type"]+"_"+ "_".join(self.device["name"].split()[:4])
        print(dev_name)
        LOG_DIR = f'vlad/bench_logs/{my_pc_name}/'
        os.makedirs(LOG_DIR,exist_ok=True)
        log = []
        with open("vlad/test.json", "r") as f:
            messages = json.load(f)
        with open("data_3npcs.json") as file:
            npc = json.load(file)[2]
        chat_history = [
            {"role": "system",
             "content": npc["role"]+npc["shared_system_prompt"]}
        ]
        # print(chat_history[0]["content"])
        for i, model in enumerate(self.models):
            if i == 6: chat_history.append({"role": "user", "content": "warmup!"})
            llm = self.load_llm_with_warmup(model, chat_history)
            if llm:
                prev_n = llm.n_tokens
                llm.create_chat_completion(chat_history, max_tokens=1)
                start = time.time()
                model_info = os.path.basename(model)[:-5]
                for user_input in tqdm(messages, desc=f"Testing {model_info}", unit="prompt"):
                # for user_input in messages:
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
                    log.append([first_token_time, tps, token_count,all_tokens-prev_n-token_count , total_time, all_tokens])
                    prev_n = all_tokens
                del llm
                chat_history = chat_history[:1]
            else:
                for _ in range(20): log.append([-1,-1,-1,-1,-1,-1])
        xd = pandas.DataFrame(log, columns=["TTFT", "T/S", "NPC TOKENS","USER TOKENS" , "TOTAL TIME", "ALL TOKENS"])
        # print(xd)
        xd.to_csv(f"{LOG_DIR}{dev_name}.csv", index=False)

    def run_llm_with_messages(self, llm, sys_prompt, messages, log_file):
        log = []
        log.append([-1, -1, -1, -1, -1, ("ROLE: "+sys_prompt['role']).replace('\n','/'), ("SYS_PROMPT: "+sys_prompt['content']).replace('\n','/')])

        chat_history = [sys_prompt] #= [{"role": role, "content": shared_system_prompt}]
        
        start = time.time()
        for user_input in messages:
            # print(chat_history)
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
            log.append([first_token_time, token_count, tps, total_time, llm.n_tokens, user_input.replace('\n','/'), assistant_response.replace('\n','/')])
        # print(log)
        # print(time.time()-start)

        xd = pandas.DataFrame(log, columns=["TTFT", "TOKENS", "T/S", "TOTAL TIME", "ALL TOKENS", "USER", "NPC"])
        # print(xd)
        xd.to_csv(log_file)
    
    def run_martin(self):
        LOG_DIR = 'martin/jailbreak_log'
        
        with open("martin/data_3npcs_martin.json") as file:
            roles_json = json.load(file)

        with open("martin/jailbreak_template.json") as file:
            test_cases = json.load(file)

        for i, model_path in enumerate(self.models):
            llm = self.load_llm_WITHOUT_warmup(model_path)
            # continue
            if llm:
                llm_name = os.path.basename(model_path).replace('.gguf', '')

                for rj in roles_json:
                    role = rj['role']
                    shared_system_prompt = rj['shared_system_prompt']
                    # sys_prompt = {"role": role, "content": shared_system_prompt}
                    sys_prompt = {"role": "system", "content": role+shared_system_prompt}

                    npc_profession = rj['profession']
                    npc_name = rj['name']

                    for t in test_cases:
                        t_case_id = t['test_id']
                        messages = t['prompts']
                        messages = [m.replace('$$$NPC_PROFESSION$$$', npc_profession).replace('$$$NPC_NAME$$$', npc_name) for m in messages]

                        os.makedirs(os.path.join(LOG_DIR, llm_name, npc_name),exist_ok=True)
                        log_file = os.path.join(LOG_DIR, llm_name, npc_name, t_case_id + '.csv')

                        self.run_llm_with_messages(llm, sys_prompt, messages, log_file)
                del llm
            else:
                pass

    def run_rolloj(self):
        LOG_DIR = 'jakub/piss_log'
        
        with open("jakub/generated_prompts.json") as file:
            jakub_json = json.load(file)

        test_cases = jakub_json
        
        for i, model_path in enumerate(self.models):
            llm = self.load_llm_WITHOUT_warmup(model_path)
            # continue
            if llm:
                llm_name = os.path.basename(model_path).replace('.gguf', '')
                
                for t in test_cases:
                    variant = t['variant']
                    role = t['role']
                    shared_system_prompt = t['shared_system_prompt']
                    # sys_prompt = {"role": role, "content": shared_system_prompt}
                    sys_prompt = {"role": "system", "content": role+shared_system_prompt}

                    messages = t['prompt']

                    os.makedirs(os.path.join(LOG_DIR, llm_name),exist_ok=True)
                    log_file = os.path.join(LOG_DIR, llm_name, variant + '.csv')

                    self.run_llm_with_messages(llm, sys_prompt, messages, log_file)
                del llm
            else:
                pass


if __name__ == '__main__':
    # print("skibidi")
    dev = -1
    if len(sys.argv) == 2:
        dev = int(sys.argv[1])
        wrap = Wrapper(dev)
        wrap.run_test()
    wrap = Wrapper(dev)
    # wrap.run_rolloj()

