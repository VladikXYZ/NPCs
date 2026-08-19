import os
import sys
import json
import platform
import questionary
from llama_cpp import Llama

import utils
from utils import get_devices, get_models, MyException

RED = "\033[91m"
GREEN = "\033[92m"
BLUE = "\033[94m"
RESET = "\033[0m"

CONTEXT_SIZE = 4096
MAX_TOKENS = 1024  # Increased for normal chatting
CUSTOM_JINJA = True


class InteractiveChat:
    def __init__(self):
        self.devices = get_devices()
        self.models = get_models()

        if not self.models:
            sys.exit(f"{RED}No models found in the models directory!{RESET}")

        # 1. Select Device
        device_choices = [f"{i} | {d['type']:<8} | {d['name']}" for i, d in enumerate(self.devices)]
        dev_choice = questionary.select("Select device:", choices=device_choices, qmark="🎮").ask()
        if not dev_choice: sys.exit("Exiting...")
        dev_idx = int(dev_choice.split("|")[0])
        self.device = self.devices[dev_idx]

        # 2. Select Model
        # print(self.models)
        model_choices = [f"{i} | {m['name']}" for i, m in enumerate(self.models)]
        mod_choice = questionary.select("Select model:", choices=model_choices, qmark="🧠").ask()
        if not mod_choice: sys.exit("Exiting...")
        mod_idx = int(mod_choice.split("|")[0])
        self.model = self.models[mod_idx]

        # 3. Apply Hardware Settings
        print(f"\n⚡ Booting {self.model['name']} on {self.device['name']}...")
        self.gpu_layers = -1 if self.device["type"] == "Vulkan" else 0
        os.environ["GGML_VK_VISIBLE_DEVICES"] = str(self.device["id"] * (self.device["type"] == "Vulkan"))

        self._start_chat()

    def _start_chat(self):
        llm_kwargs = {
            "model_path": self.model["path"],
            "n_gpu_layers": self.gpu_layers,
            "n_ctx": CONTEXT_SIZE,
            "verbose": False,
            "temperature": 0.7  # Better for generic chatting than 0
        }

        # Setup simple system prompt and warmup
        system_prompt = "You are a helpful, smart, and concise AI assistant."
        chat_history = [{"role": "system", "content": system_prompt}]
        warmup = chat_history + [{"role": "user", "content": "Hello"}]

        # Load using your utils function
        try:
            llm = utils.load_llm(self.model, llm_kwargs, custom_jinja=CUSTOM_JINJA, log=True)
            if not llm:
                sys.exit(f"\n{RED}Failed to load the model.{RESET}")
        except Exception as e:
            sys.exit(f"\n{RED}Error loading model: {e}{RESET}")

        print(f"\n{GREEN}========================================={RESET}")
        print(f"{GREEN} Chat Session Started! Type '/exit' to quit{RESET}")
        print(f"{GREEN}========================================={RESET}")

        while True:
            try:
                # Get user input
                user_input = input(f"\n{GREEN}You:{RESET} ")

                # Check for exit commands
                if user_input.strip().lower() in ['/exit', '/quit']:
                    print("Ending session...")
                    break
                if not user_input.strip():
                    continue

                # Add to history
                chat_history.append({"role": "user", "content": user_input})
                print(f"{BLUE}Assistant:{RESET} ", end="", flush=True)

                # Stream the generation
                assistant_response = ""
                stream = llm.create_chat_completion(messages=chat_history, stream=True, max_tokens=MAX_TOKENS)

                for chunk in stream:
                    delta = chunk['choices'][0]["delta"]
                    if 'content' in delta:
                        content = delta['content']
                        assistant_response += content
                        print(content, end="", flush=True)

                print()  # Newline after generation finishes

                # Append assistant's response to history
                chat_history.append({"role": "assistant", "content": assistant_response})

            except KeyboardInterrupt:
                # Allows you to press Ctrl+C to cleanly exit the chat
                print("\nExiting chat...")
                break
            except Exception as e:
                print(f"\n{RED}Inference Error: {e}{RESET}")
                print("\nExiting chat...")
                break

        if hasattr(llm, 'close'):
            llm.close()
        del llm


if __name__ == '__main__':
    InteractiveChat()