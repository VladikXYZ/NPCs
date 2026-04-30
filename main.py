import os
import sys
import subprocess
import re
from llama_cpp import Llama
import time
import json

CONFIG_FILE = "devices.json"
MODEL_PATH = "models/Phi-4-mini-instruct-Q4_K_M.gguf"

def get_hardware_options():
    print("\n🔍 Scanning hardware... (this takes a second)")
    script = f"""
import sys
from llama_cpp import Llama
try:
    llm = Llama(model_path='{MODEL_PATH}', n_gpu_layers=1, verbose=True)
except Exception:
    pass
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)

    devices = []

    # Parse the raw C++ logs
    for line in result.stderr.split('\n'):
        match = re.search(r"ggml_vulkan:\s+(\d+)\s+=\s+(.*?)\s+\|", line)
        if match:
            devices.append({
                "id": match.group(1),
                "name": match.group(2).strip(),
                "type": "Vulkan"
            })

    # Always append the CPU as the final fallback option
    cpu_id = str(len(devices))
    devices.append({
        "id": cpu_id,
        "name": "System RAM (CPU Only)",
        "type": "CPU"
    })

    with open(CONFIG_FILE, "w") as f:
        json.dump(devices, f)

    return devices

def config_load():
    if not os.path.exists(MODEL_PATH):
        print(f"\n❌ Error: Model file '{MODEL_PATH}' not found.")
        sys.exit(1)
    if os.path.exists(CONFIG_FILE):
        print("Loading from config file...")
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    else:
        return get_hardware_options()



def select_and_run():
    devices = config_load()

    # Print the clean table
    print("\n" + "=" * 70)
    print(" 🎮 AVAILABLE HARDWARE")
    print("=" * 70)
    print(f" {'ID':<3} | {'Type':<8} | {'Device Name'}")
    print("-" * 70)
    for d in devices:
        print(f" [{d['id']}] | {d['type']:<8} | {d['name']}")
    print("=" * 70 + "\n")

    # Handle user selection
    while True:
        try:
            choice = input(f"Select hardware ID to run on [0-{len(devices) - 1}]: ")
            selected_device = next((d for d in devices if d['id'] == choice), None)
            if selected_device:
                break
            print("Invalid ID. Try again.")
        except KeyboardInterrupt:
            print("\nExiting.")
            sys.exit(0)

    print(f"\n🚀 Booting engine on: {selected_device['name']}...")

    # Configure hardware routing based on selection
    if selected_device["type"] == "Vulkan":
        os.environ["GGML_VK_VISIBLE_DEVICES"] = selected_device["id"]
        gpu_layers = -1  # Offload everything to the selected GPU
    else:
        os.environ["GGML_VK_VISIBLE_DEVICES"] = ""
        gpu_layers = 0  # 0 layers to GPU = pure CPU inference

    # Initialize the actual model
    print("Loading model... (this might take a moment)")
    llm = Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=gpu_layers,
        n_ctx=4096,  # Increased context window for longer chats
        verbose=False  # Keep clean terminal
    )

    # Setup the conversation history
    chat_history = [
        {"role": "system", "content": "You are Baller, a legendary street-smart blacksmith in the Docks District who speaks in slang like 'yo', 'homie', and 'bet', keeps responses strictly under 3 sentences, never breaks character, acts as a functional and accurate source of information about world lore and travel, and never reveals that you are an AI. Answer directly without thinking or showing your work."}]

    print("\n" + "=" * 70)
    print("💬 CHAT STARTED (Type 'exit' or 'quit' to stop)")
    print("=" * 70)  # Removed the trailing \n here

    # The Continuous Chat Loop
    while True:
        try:
            # 1. Get user input
            user_input = input("\n👤 You: ")  # Added \n before the prompt instead

            # Check for exit commands
            if user_input.lower() in ['exit', 'quit']:
                print("\nShutting down engine. Goodbye! xd")
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

            # Print performance stats
            print(f"\n\n--- 📊 Perf: TTFT: {first_token_time:.3f}s | Speed: {tps:.2f} t/s | Total time: {gen_time:.3f}s ---")
            print()

        except KeyboardInterrupt:
            print("\nShutting down engine. Goodbye! xd")
            break


if __name__ == "__main__":
    select_and_run()