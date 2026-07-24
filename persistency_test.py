import time
import json
import pandas as pd
from tqdm import tqdm
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Jinja2ChatFormatter # <-- Add this import

MODEL_DIR = "models/"
MODEL_NAME = "Qwen3.5-0.8B-Q4_K_M"
TEMP = """
{%- set shared_prompt = "You are a fantasy RPG NPC. Speak ONLY pure dialogue with NO stage directions, actions, or asterisks. Be direct and terse. Answer the player's exact question and immediately stop talking. Do NOT volunteer background facts unless directly asked, and do NOT over-explain. Treat your reality as a normal fantasy world. Maximum length: 2 short sentences." -%}

{# 1. Render the Persona and Shared Prompt #}
{%- if messages and messages[0].role == 'system' -%}
    {{- '<|im_start|>system\n' + messages[0].content + '\n\n' + shared_prompt + '<|im_end|>\n' -}}
{%- else -%}
    {{- '<|im_start|>system\n' + shared_prompt + '<|im_end|>\n' -}}
{%- endif -%}

{# 2. Loop through the chat history #}
{%- for message in messages -%}
    {%- if message.role != 'system' -%}
        {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' -}}
    {%- endif -%}
{%- endfor -%}

{# 3. Prompt the model to generate the next response #}
{%- if add_generation_prompt -%}
    {{- '<|im_start|>assistant\n<think>\n\n</think>\n' -}}
{%- endif -%}
"""
my_custom_handler = Jinja2ChatFormatter(
    template=TEMP,
    eos_token="<|im_end|>",
    bos_token=""
).to_chat_handler()

print(f"⚡ Loading {MODEL_NAME} directly into memory...")
llm = Llama(
    model_path=f"{MODEL_DIR}{MODEL_NAME}.gguf",
    n_ctx=8192,
    n_gpu_layers=-1,
    verbose=False,
    # chat_format="chatml"
    chat_handler=my_custom_handler
)

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

system_prompt = NPC["role"] + "\n" + NPC["shared_system_prompt"]

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
    chat_history.append({"role": "assistant", "content": f"<think>\n\n</think>\n{output_text}"})