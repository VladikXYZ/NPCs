import os
import sys
import time

import pandas
from llama_cpp import Llama

from setup import Options


def chat(device, model, role = ""):
    print(f"Using {model} on {device["name"]}")
    if device["type"] == "Vulkan":
        os.environ["GGML_VK_VISIBLE_DEVICES"] = device["id"]
        gpu_layers = -1
    else:
        os.environ["GGML_VK_VISIBLE_DEVICES"] = ""
        gpu_layers = 0

    # Initialize the actual model
    print("Loading model... (this might take a moment)")
    try:
        llm = Llama(model_path=model, n_gpu_layers=gpu_layers, n_ctx=4096, verbose=False)
    except Exception as e:
        # print(f"Failed to load model: {e}")
        print("Not enough memory")
        return

    # Setup the conversation history
    chat_history = [
        {"role": "system",
         "content": "You are Baller, a legendary street-smart blacksmith in the Docks District who speaks in slang like 'yo', 'homie', and 'bet', keeps responses strictly under 3 sentences, never breaks character, acts as a functional and accurate source of information about world lore and travel, and never reveals that you are an AI. Answer directly without thinking or showing your work."}]

    print("\n" + "=" * 70)
    print("💬 CHAT STARTED (Type 'exit' or 'quit' to stop)")
    print("=" * 70)  # Removed the trailing \n here

    log = []

    # The Continuous Chat Loop
    while True:
        try:
            # 1. Get user input
            user_input = input("\n👤 You: ")  # Added \n before the prompt instead

            # Check for exit commands
            if user_input.lower() in ['exit', 'quit']:
                print("\nShutting down engine. Goodbye! xd")
                # print(chat_history)
                print(log)
                xd = pandas.DataFrame(log, columns=["TTFT", "TOKENS", "T/S", "TOTAL TIME"], index=None)
                print(xd)
                break
            if not user_input.strip():
                continue

            # 2. Add user message to history
            chat_history.append({"role": "user", "content": user_input})

            # ... (inside the while True loop, right before llm.create_chat_completion)

            print("🤖 AI:  ", end="", flush=True)

            # Timer start
            start_time = time.perf_counter()
            first_token_time = None
            token_count = 0

            # 3. Stream the response
            stream = llm.create_chat_completion(messages=chat_history, stream=True, max_tokens=128)

            assistant_response = ""
            for chunk in stream:
                delta = chunk['choices'][0].get('delta', {})
                if 'content' in delta:
                    # Capture TTFT (Time To First Token)
                    if first_token_time is None:
                        first_token_time = time.perf_counter() - start_time

                    text = delta['content']
                    print(text, end="", flush=True)
                    assistant_response += text
                    token_count += 1

            # End timer
            total_time = time.perf_counter() - start_time
            # Speed excluding the initial delay (TTFT)
            gen_time = total_time - (first_token_time if first_token_time else 0)
            tps = token_count / gen_time if gen_time > 0 else 0

            # 4. Save the AI's response to history
            chat_history.append({"role": "assistant", "content": assistant_response})
            log.append([first_token_time, token_count, tps, total_time])

            # Print performance stats
            print(
                f"\n\n--- 📊 Perf: TTFT: {first_token_time:.3f}s | Speed: {tps:.2f} t/s | Total tokens: {token_count} | Total time: {gen_time:.3f}s | Tokens so far: {llm.n_tokens} ---")
            print()

        except KeyboardInterrupt:
            print("\nShutting down engine. Goodbye! xd")
            break


if __name__ == "__main__":
    opt = Options()
    params = [len(opt.devices), len(opt.models)]
    infos = [opt.devices_info, opt.models_info]
    pos = 0
    if len(sys.argv) == 3:
        for arg in sys.argv[1:]:
            val = arg
            while True:
                try:
                    val = int(val)
                    if 0 <= val < params[pos]:
                        params[pos] = val
                        pos += 1
                        break
                    else:
                        infos[pos]()
                        val = input("Enter valid value: ")
                except:
                    infos[pos]()
                    val = input("Enter valid value: ")
        chat(opt.device(params[0]), opt.model(params[1]))

