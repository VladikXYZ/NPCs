import time
import json

import pandas
import pandas as pd
from tqdm import tqdm
from llama_cpp import Llama
from models.templating import get_handler

MODEL_DIR = "models/"
MODEL = "Qwen3.5-9B-Q4_K_M"
# MODEL = "Bonsai-27B-Q1_0"
llm_kwargs = {
    "model_path": "models/" + MODEL + ".gguf",
    "n_gpu_layers": -1,
    "n_ctx": 256*16,
    "verbose": False
}
# print(model, end="")
my_custom_handler = get_handler(MODEL)
if my_custom_handler:
    print("Using custom")
    llm_kwargs["chat_handler"] = my_custom_handler

print(f"⚡ Loading {MODEL} directly into memory...")
llm = Llama(**llm_kwargs)
# print(llm.metadata.get("tokenizer.chat_template"))
# print(llm.chat_handler)
# --- 🕵️ THE INTERCEPTOR (Monkey-Patch) ---
original_create_completion = llm.create_completion
debug_prompts = []

def spy_create_completion(*args, **kwargs):
    # Extract the string prompt or token list passed by the chat handler
    prompt_data = kwargs.get("prompt", args[0] if len(args) > 0 else None)
    
    # If it's a string, tokenize it exactly as the engine will
    if isinstance(prompt_data, str):
        prompt_tokens = llm.tokenize(prompt_data.encode("utf-8"))
        prompt_string = prompt_data
    else:
        prompt_tokens = prompt_data
        prompt_string = llm.detokenize(prompt_tokens).decode("utf-8", errors="ignore")

    debug_prompts.append((prompt_string, prompt_tokens))
    turn_num = len(debug_prompts)
    
    if turn_num > 1:
        prev_string, prev_tokens = debug_prompts[-2]
        
        # 1. Check String Mismatch
        if prompt_string.startswith(prev_string):
            print(f"\n[Turn {turn_num}] ✅ STRING CHECK: The new prompt perfectly extends the previous text.")
        else:
            print(f"\n[Turn {turn_num}] ❌ STRING MISMATCH:")
            for i in range(min(len(prev_string), len(prompt_string))):
                if prev_string[i] != prompt_string[i]:
                    print(f"   -> Diverges at character index {i}:")
                    print(f"      Old: {repr(prev_string[max(0, i-15):i+15])}")
                    print(f"      New: {repr(prompt_string[max(0, i-15):i+15])}")
                    break

        # 2. Check Token Mismatch (This is what actually breaks the cache!)
        old_len = len(prev_tokens)
        if prompt_tokens[:old_len] == prev_tokens:
            print(f"[Turn {turn_num}] ✅ TOKEN CHECK: Token IDs match perfectly. Cache HIT should occur.")
        else:
            print(f"[Turn {turn_num}] ❌ TOKEN MISMATCH: The prefix tokens changed! Cache WILL FAIL.")
            for i in range(min(old_len, len(prompt_tokens))):
                if prev_tokens[i] != prompt_tokens[i]:
                    print(f"   -> Diverges at token index {i}:")
                    print(f"      Old Token ID: {prev_tokens[i]} -> {repr(llm.detokenize([prev_tokens[i]]).decode('utf-8'))}")
                    print(f"      New Token ID: {prompt_tokens[i]} -> {repr(llm.detokenize([prompt_tokens[i]]).decode('utf-8'))}")
                    break
                    
    return original_create_completion(*args, **kwargs)

# Bind our spy function to the LLM
llm.create_completion = spy_create_completion
# -----------------------------------------

with open("vlad/test.json", "r") as f: 
    MESSAGES = json.load(f)
with open("data_3npcs.json", "r") as file: 
    NPC = json.load(file)[2]

system_prompt = NPC["role"]

log_results = []
chat_history = [{"role": "system", "content": system_prompt}]

for turn_idx, user_input in enumerate(MESSAGES):
    print(f"\n--- Starting Turn {turn_idx + 1} ---")
    chat_history.append({"role": "user", "content": user_input})
    
    start_time = time.perf_counter()
    first_token_time = None
    output_text = ""
    completion_tokens = 0
    
    stream = llm.create_chat_completion(
        messages=chat_history,
        stream=True,
        max_tokens=32,
        temperature=0.0
    )
    
    for chunk in stream:
        delta = chunk["choices"][0]["delta"]
        if "content" in delta and delta["content"]:
            if first_token_time is None:
                first_token_time = time.perf_counter()
            output_text += delta["content"]
            completion_tokens += 1
            
    end_time = time.perf_counter()
# If the output text didn't include <think>, prepend it or preserve the raw response
    chat_history.append({"role": "assistant", "content": output_text})
#     chat_history.append({"role": "assistant", "content": f"<think>\n\n</think>\n\n{str(output_text)}"})

# print(chat_history)
xd = [[x["role"], x["content"]] for x in chat_history]
xd = pandas.DataFrame(xd)
xd.to_csv("templates.csv")