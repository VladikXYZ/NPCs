import time
import json
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

"""
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DGGML_VULKAN=ON
cmake --build build --config Release -j$(nproc)
"""

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed"
)

MODEL_NAME = "TTFT testing"

with open("vlad/ttft_test.json", "r") as f:
    MESSAGES = json.load(f)
with open("data_3npcs.json", "r") as file:
    NPC = json.load(file)[2]

system_prompt = NPC["role"] + "\n" + NPC["shared_system_prompt"]

log_results = []
chat_history = [{"role": "system", "content": system_prompt}]

for turn_idx, user_input in enumerate(tqdm(MESSAGES, desc="Benchmarking")):
    chat_history.append({"role": "user", "content": user_input})

    start_time = time.perf_counter()
    first_token_time = None
    output_text = ""
    completion_tokens = 0

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=chat_history,
            max_tokens=32,
            temperature=0.0,
            stream=True,
            stream_options={"include_usage": True}
        )

        for chunk in response:
            if not chunk.choices:
                if hasattr(chunk, 'usage') and chunk.usage:
                    completion_tokens = chunk.usage.completion_tokens
                continue

            delta = chunk.choices[0].delta.content
            if delta:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                output_text += delta
                completion_tokens += 1

        end_time = time.perf_counter()

    except Exception:
        break

    ttft = (first_token_time - start_time) if first_token_time else (end_time - start_time)
    generation_time = (end_time - first_token_time) if first_token_time else 0.0
    tps = completion_tokens / generation_time if generation_time > 0 else 0.0
    total_time = end_time - start_time

    log_results.append({
        "MODEL": MODEL_NAME,
        "TTFT": ttft,
        "T/s": tps,
        "TOTAL TIME": total_time,
        "COMPLETION TOKENS": completion_tokens,
        "PROMPT": user_input.replace('\n', '|'),
        "RESPONSE": output_text.replace('\n', '|')
    })

    chat_history.append({"role": "assistant", "content": output_text})

df = pd.DataFrame(log_results)
df.to_csv("bonsai.csv", index=False)