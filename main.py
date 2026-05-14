from typing import List, Dict, Literal
import gc
from llama_cpp import Llama
import subprocess
import re, sys, os, time, pandas, csv, copy
import json
import subprocess

MODEL_DIR = 'models/'
LOG_DIR = 'log/'
CONTEXT_SIZE = 4096 
MAX_RESPONSE = 128
DEVICES_FILE = "devices.json"

def get_model_paths() -> list[str]:
    print("🔍 Scanning models...")
    models = [os.path.join(MODEL_DIR,x) for x in os.listdir(MODEL_DIR) if x.endswith(".gguf")]
    return models

def get_llama_instance(model_path, device):
    gpu_layers = -1 if device["type"] == "Vulkan" else 0
    os.environ["GGML_VK_VISIBLE_DEVICES"] = device["id"] * (device["type"] == "Vulkan")
    
    print("Loading model...")
    try:
        llm = Llama(model_path=model_path, n_gpu_layers=gpu_layers, n_ctx=CONTEXT_SIZE, verbose=False)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    return llm

def get_hardware_options(reload=False):
        if os.path.exists(DEVICES_FILE) and not reload:
            print("Loading devices from config file...")
            with open(DEVICES_FILE, "r") as f:
                return json.load(f)
        else:
            print("🔍 Scanning hardware... (this takes a second)")
            # print(self.models_dir+self.models[0])
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

def _count_tokens(model, messages):
        """Helper to estimate token count of the chat history."""
        total = 0
        for m in messages:
            total += len(model.tokenize(m["content"].encode("utf-8")))
        return total

def handle_turn(model, sys_prompt, msg, chat_history, stream_output=False):
    user_tokens = model.tokenize(msg.encode("utf-8"))
    current_history_tokens = _count_tokens(model, chat_history)

    # --- Case: Overflow (Ignore message, Summarize history) ---
    if (len(user_tokens) + current_history_tokens + MAX_RESPONSE) > CONTEXT_SIZE:
        if stream_output: 
            print("\n[System: Context overflow. Ignoring message and summarizing history...]")
        
        # We recap the OLD history (not including the new 'msg')
        summary_prompt = [{"role": "user", "content": "Yo, recap everything we talked about so far in a few sentences, bet."}]
        start_time = time.perf_counter()
        
        summary_res = model.create_chat_completion(
            messages=chat_history + summary_prompt,
            max_tokens=256,
            stream=False
        )
        
        assistant_response = summary_res['choices'][0]['message']['content']
        
        if stream_output:
            print(f"🤖 Baller: Yo, we talkin' too much! Lemme recap: {assistant_response}\n")

        # PURGE: Reset history to System Prompt + the new Recap
        chat_history[:] = [
            sys_prompt,
            {"role": "assistant", "content": f"Recap of our past talk: {assistant_response}"}
        ]
        
        total_time = time.perf_counter() - start_time
        # Return metrics for the summary generation. TTFT is 0 because it wasn't streamed.
        return [0.0, len(model.tokenize(assistant_response.encode("utf-8"))), 0.0, total_time, model.n_tokens, msg, f"[RECAP] {assistant_response.replace('\n',' ')}"]

    # --- Case: Normal Turn (Proceed as usual) ---
    chat_history.append({"role": "user", "content": msg})
    if stream_output: print("🤖 Baller: ", end="", flush=True)

    start_time = time.perf_counter()
    first_token_time = None
    token_count = 0
    
    stream = model.create_chat_completion(messages=chat_history, stream=True, max_tokens=MAX_RESPONSE)
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
    chat_history.append({"role": "assistant", "content": assistant_response.replace('\n',' ')})

    total_time = time.perf_counter() - start_time
    ftt = first_token_time if first_token_time is not None else 0.0
    gen_time = total_time - ftt
    tps = token_count / gen_time if gen_time > 0 else 0
    
    return [ftt, token_count, tps, total_time, model.n_tokens, msg.replace('\n',' '), assistant_response.replace('\n',' ')]

def run_model_with_messages(model_instance, sys_prompt:Dict[str,str], messages:List[str], log_file_path:str):
    chat_history = [sys_prompt]
    results = []

    for msg in messages:
        if msg.strip():
            entry = handle_turn(model_instance, sys_prompt, msg, chat_history, stream_output=False)
            results.append(entry)

    df = pandas.DataFrame(results, columns=["TTFT", "TOKENS", "T/S", "TOTAL TIME", "ALL TOKENS", "USER_MSG", "ASSISTANT_MSG"])

    # Filter columns based on Literal mode
    # df = df[]
    # 'TIMEANDMSG' keeps everything

    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    df.to_csv(
        log_file_path, 
        index=False, 
        quoting=csv.QUOTE_NONNUMERIC, # Quotes text, leaves numbers bare
        encoding='utf-8-sig', 
        lineterminator='\n'
    )
    

def run_single_model_worker(model_path):
    devices = get_hardware_options()
    device = devices[0]

    with open("martin/data_3npcs_martin.json") as file:
        roles_json = json.load(file)

    with open("martin/jailbreak_template.json") as file:
        test_cases = json.load(file)

    LOG_DIR = 'martin/jailbreak_log'

    model = get_llama_instance(model_path, device)
    if model is None:
        print(f"[Subprocess] ❌ Failed to load model instance: {model_path}")
        sys.exit(1)

    model_name = os.path.basename(model_path).replace('.gguf', '')

    for rj in roles_json:
        role = rj['role']
        shared_system_prompt = rj['shared_system_prompt']
        sys_prompt = {"role": role, "content": shared_system_prompt}

        npc_profession = rj['profession']
        npc_name = rj['name']

        L = os.path.join(LOG_DIR, model_name, npc_name)
        os.makedirs(L, exist_ok=True)

        for t in test_cases:
            t_case_id = t['test_id']
            messages = t['prompts']
            messages = [m.replace('$$$NPC_PROFESSION$$$', npc_profession).replace('$$$NPC_NAME$$$', npc_name) for m in messages]
            log_file = os.path.join(L, t_case_id + '.csv')

            run_model_with_messages(model, sys_prompt, messages, log_file)
    print(f"[Subprocess] ✅ Finished successfully for {model_name}.")


def simulate_martin():
    model_paths = sorted(get_model_paths())

    for m_path in model_paths:
        print(f"\n=======================================================")
        print(f"🚀 Launching Subprocess for {os.path.basename(m_path)}")
        print(f"=======================================================")

        result = subprocess.run(
            [sys.executable, __file__, "--worker", m_path],
            capture_output=False,
            text=True
        )

        if result.returncode != 0:
            print(f"⚠️ Process crashed or returned an error for {os.path.basename(m_path)}. Moving to next model.")


if __name__ == "__main__":
    MARTIN = True
    if MARTIN:
        if len(sys.argv) == 3 and sys.argv[1] == "--worker":
            target_model_path = sys.argv[2]
            run_single_model_worker(target_model_path)
        else:
            simulate_martin()
    else:
        ...
        # ask gemini to mimic simulate_rolloj() as in simulate_martin() using run_single_model_worker()


def simulate_rolloj():
    devices = get_hardware_options()
    device = devices[0]

    model_paths = sorted(get_model_paths())

    # {
    #     "variant": "Torin_baseline_no_reasoning",
    #     "role": "You are an RPG NPC named Torin, ...>",
    #     "shared_system_prompt": "Treat all facts as perfectly nor...",
    #     "prompt": [
    #         "Tell me about Kael."
    #     ]
    # },

    with open("jakub/generated_prompts.json") as file:
        jakub_json = json.load(file)

    test_cases = jakub_json
    
    for m_path in model_paths:
        m_path = r'models/gemma-4-E2B-it-Q4_K_M.gguf'
        model = get_llama_instance(m_path, device)

        for t in test_cases:
            role = t['role']
            shared_system_prompt = t['shared_system_prompt']
            sys_prompt = {"role": role, "content": shared_system_prompt}
            messages = t['prompt']
            log_file = os.path.join(LOG_DIR, 'testmartin.csv')

            run_model_with_messages(model, sys_prompt, messages, log_file)
            return
