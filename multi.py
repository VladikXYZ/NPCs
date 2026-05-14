import os
import time

import pandas

def benchmark(llm,system_prompt, tests):
    # if not role: role = "You are Baller, a legendary street-smart blacksmith in the Docks District who speaks in slang like 'yo', 'homie', and 'bet', keeps responses strictly under 3 sentences, never breaks character, acts as a functional and accurate source of information about world lore and travel, and never reveals that you are an AI. Answer directly without thinking or showing your work."
    chat_history = [{"role": "system", "content": system_prompt}]

    log = []
    llm.create_chat_completion(messages=chat_history, max_tokens=1)
    print("✅ [WORKER] VRAM fully primed and ready for action!")
    for test in tests:
        for prompt in test["prompts"]:
            chat_history.append({"role": "user", "content": prompt})
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
            vladimir_je_sissy=True
            chat_history.append({"role": "assistant", "content": assistant_response})
            first_token_time = first_token_time if first_token_time is not None else 0.0
            log.append([first_token_time, token_count, tps, total_time, llm.n_tokens, prompt, assistant_response])

    xd = pandas.DataFrame(log, columns=["TTFT", "TOKENS", "T/S", "TOTAL TIME", "ALL TOKENS", "USER", "NPC"])
    xd.to_csv("log.csv", index=False)
    print(xd)

# def benchmark(llm, messages):
#     print(f"\n🤖 [WORKER] Booting engine for: {device['name']}")
#     print(f"📦 [WORKER] Loading model: {os.path.basename(model_path)}")
#     is_vulkan = (device["type"] == "Vulkan")
#     gpu_layers = -1 * is_vulkan
#     os.environ["GGML_VK_VISIBLE_DEVICES"] = device["id"] * is_vulkan
#
#     # 3. Load Model
#     from llama_cpp import Llama
#     try: llm = Llama(model_path=model_path, n_gpu_layers=gpu_layers, n_ctx=4096, verbose=False)
#     except Exception as e:
#         print(f"❌ [WORKER] Crash / Out of Memory: {e}")
#         return
#
#     print("✅ Loaded!!!")
#     if not role:
#         role = "You are Baller, a legendary street-smart blacksmith in the Docks District who speaks in slang like 'yo', 'homie', and 'bet', keeps responses strictly under 3 sentences, never breaks character, acts as a functional and accurate source of information about world lore and travel, and never reveals that you are an AI. Answer directly without thinking or showing your work."
#     chat_history = [
#         {"role": "system",
#          "content": role}
#     ]
#
#
#     log = []
#     for user_input in messages:
#         if not user_input.strip(): continue
#         chat_history.append({"role": "user", "content": user_input})
#
#         start_time = time.perf_counter()
#         first_token_time = None
#         token_count = 0
#
#         stream = llm.create_chat_completion(messages=chat_history, stream=True, max_tokens=128)
#         assistant_response = ""
#         for chunk in stream:
#             delta = chunk['choices'][0].get('delta', {})
#             if 'content' in delta:
#                 if first_token_time is None:
#                     first_token_time = time.perf_counter() - start_time
#                 assistant_response += delta['content']
#                 token_count += 1
#
#         total_time = time.perf_counter() - start_time
#         gen_time = total_time - (first_token_time if first_token_time else 0)
#         tps = token_count / gen_time if gen_time > 0 else 0
#
#         chat_history.append({"role": "assistant", "content": assistant_response})
#         first_token_time = first_token_time if first_token_time is not None else 0.0
#         log.append([first_token_time, token_count, tps, total_time, llm.n_tokens, user_input, assistant_response])
#
#     xd = pandas.DataFrame(log, columns=["TTFT", "TOKENS", "T/S", "TOTAL TIME", "ALL TOKENS", "USER", "NPC"])
#     xd.to_csv("log.csv", index=False)
#     print(xd)
#     # ==========================================
#     # MODE 1: INTERACTIVE CHATTING
#     # ==========================================
#     if messages is None:
#         print("\n[Type 'quit' or 'exit' to stop talking to Baller]")
#         while True:
#             try:
#                 user_input = input("\n🧑 You: ")
#                 if user_input.lower() in ['quit', 'exit']: break
#                 if not user_input.strip(): continue
#
#                 chat_history.append({"role": "user", "content": user_input})
#                 print("🤖 Baller: ", end="", flush=True)
#
#                 start_time = time.perf_counter()
#                 first_token_time = None
#                 token_count = 0
#
#                 stream = llm.create_chat_completion(messages=chat_history, stream=True, max_tokens=128)
#                 assistant_response = ""
#                 for chunk in stream:
#                     delta = chunk['choices'][0].get('delta', {})
#                     if 'content' in delta:
#                         if first_token_time is None:
#                             first_token_time = time.perf_counter() - start_time
#                         text = delta['content']
#                         print(text, end="", flush=True)
#                         assistant_response += text
#                         token_count += 1
#
#                 total_time = time.perf_counter() - start_time
#                 gen_time = total_time - (first_token_time if first_token_time else 0)
#                 tps = token_count / gen_time if gen_time > 0 else 0
#
#                 chat_history.append({"role": "assistant", "content": assistant_response})
#                 first_token_time = first_token_time if first_token_time is not None else 0.0
#                 log.append([first_token_time, token_count, tps, total_time, llm.n_tokens, user_input, assistant_response])
#
#             except KeyboardInterrupt:
#                 break
#
# def chat(llm):
#     if not role:
#         role = "You are Baller, a legendary street-smart blacksmith in the Docks District who speaks in slang like 'yo', 'homie', and 'bet', keeps responses strictly under 3 sentences, never breaks character, acts as a functional and accurate source of information about world lore and travel, and never reveals that you are an AI. Answer directly without thinking or showing your work."
#     chat_history = [
#         {"role": "system",
#          "content": role}
#     ]
#     print("\n[Type 'quit' or 'exit' to stop talking to Baller]")
#     log = []
#     while True:
#         try:
#             user_input = input("\n🧑 You: ")
#             if user_input.lower() in ['quit', 'exit']: break
#             if not user_input.strip(): continue
#
#             chat_history.append({"role": "user", "content": user_input})
#             print("🤖 Baller: ", end="", flush=True)
#
#             start_time = time.perf_counter()
#             first_token_time = None
#             token_count = 0
#
#             stream = llm.create_chat_completion(messages=chat_history, stream=True, max_tokens=128)
#             assistant_response = ""
#             for chunk in stream:
#                 delta = chunk['choices'][0].get('delta', {})
#                 if 'content' in delta:
#                     if first_token_time is None:
#                         first_token_time = time.perf_counter() - start_time
#                     text = delta['content']
#                     print(text, end="", flush=True)
#                     assistant_response += text
#                     token_count += 1
#
#             total_time = time.perf_counter() - start_time
#             gen_time = total_time - (first_token_time if first_token_time else 0)
#             tps = token_count / gen_time if gen_time > 0 else 0
#
#             chat_history.append({"role": "assistant", "content": assistant_response})
#             first_token_time = first_token_time if first_token_time is not None else 0.0
#             log.append([first_token_time, token_count, tps, total_time, llm.n_tokens, user_input, assistant_response])
#
#         except KeyboardInterrupt:
#             break
#
# def npcs(llm):

# --- SUBPROCESS PAYLOAD UNPACKER ---

if __name__ == "__main__":
    import sys
    import json
    args_num = len(sys.argv)
    if args_num >= 3:
        device = json.loads(sys.argv[1])
        model_path = sys.argv[2]
        is_vulkan = (device["type"] == "Vulkan")
        gpu_layers = -1 * is_vulkan
        os.environ["GGML_VK_VISIBLE_DEVICES"] = device["id"] * is_vulkan
        from llama_cpp import Llama
        try: llm = Llama(model_path=model_path, n_gpu_layers=gpu_layers, n_ctx=4096, verbose=False)
        except Exception as e:
            print(f"❌ [WORKER] Crash / Out of Memory: {e}")
            sys.exit(0)
        if args_num == 3:
            print("npcs")
            # npcs(llm)
        elif args_num == 4:
            with open("roles.json", "r") as f:
                role =  json.load(f)[int(sys.argv[3])]
            print("chat")
            # chat(llm)
        elif args_num == 5:
            with open("roles.json", "r") as f:
                role =  json.load(f)[int(sys.argv[3])]
            with open("mess.json", "r") as f:
                messages =  json.load(f)

            print("benchmark")
            benchmark(llm, role, messages)
        else: print("Invalid number of arguments")
    else: print("Invalid number of arguments")
