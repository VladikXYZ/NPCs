import json
import os
import sys
import time
import pandas
from llama_cpp import Llama
from setup import Options

def bench(device, model, messages):
    print(f"Using {model} on {device["name"]}")
    if device["type"] == "Vulkan":
        os.environ["GGML_VK_VISIBLE_DEVICES"] = device["id"]
        gpu_layers = -1
    else:
        os.environ["GGML_VK_VISIBLE_DEVICES"] = ""
        gpu_layers = 0

    # Initialize the actual model
    print("Loading model... (this might take a moment)")
    llm = Llama(model_path=model, n_gpu_layers=gpu_layers,n_ctx=4096, verbose=False)
    print("Loaded!!!")
    chat_history = [
        {"role": "system",
         "content": "You are Baller, a legendary street-smart blacksmith in the Docks District who speaks in slang like 'yo', 'homie', and 'bet', keeps responses strictly under 3 sentences, never breaks character, acts as a functional and accurate source of information about world lore and travel, and never reveals that you are an AI. Answer directly without thinking or showing your work."}]


    log = []
    for user_input in messages:
        if not user_input.strip():
            continue

            # 1. Print the simulated user input
        # print(f"\n👤 You: {user_input}")

        # 2. Add user message to history
        chat_history.append({"role": "user", "content": user_input})

        # print("🤖 AI:  ", end="", flush=True)

        # Timer start
        start_time = time.perf_counter()
        first_token_time = None
        token_count = 0

        # 3. Stream the response
        stream = llm.create_chat_completion(
            messages=chat_history,
            stream=True,
            max_tokens=1024
        )

        assistant_response = ""
        for chunk in stream:
            delta = chunk['choices'][0].get('delta', {})
            if 'content' in delta:
                # Capture TTFT (Time To First Token)
                if first_token_time is None:
                    first_token_time = time.perf_counter() - start_time

                text = delta['content']
                # print(text, end="", flush=True)
                assistant_response += text
                token_count += 1

        # End timer
        total_time = time.perf_counter() - start_time
        # Speed excluding the initial delay (TTFT)
        gen_time = total_time - (first_token_time if first_token_time else 0)
        tps = token_count / gen_time if gen_time > 0 else 0

        # 4. Save the AI's response to history and log metrics
        chat_history.append({"role": "assistant", "content": assistant_response})

        # Guard against empty responses preventing a crash on None
        first_token_time = first_token_time if first_token_time is not None else 0.0
        log.append([first_token_time, token_count, tps, total_time])

        # Print performance stats for this specific prompt
        # print(f"\n\n--- 📊 Perf: TTFT: {first_token_time:.3f}s | Speed: {tps:.2f} t/s | Total tokens: {token_count} | Total time: {gen_time:.3f}s | Tokens so far: {llm.n_tokens} ---")

        # ---------------------------------------------------------
        # THIS RUNS AFTER ALL MESSAGES ARE PROCESSED
        # ---------------------------------------------------------
    # print("\n" + "=" * 70)
    # print("🏁 BENCHMARK COMPLETE")
    # print("=" * 70)

    # print(log)
    xd = pandas.DataFrame(log, columns=["TTFT", "TOKENS", "T/S", "TOTAL TIME"], index=None)
    print(xd)


if __name__ == "__main__":
    opt = Options()
    with open("messages.json", "r") as f:
        messages = json.load(f)
    # print(messages)
    for device in opt.devices[:2]:
        # print(device)
        for i in range(len(opt.models)):
            model = opt.model(i)
            bench(device, model, messages)
            # print(model)