import sys

if __name__ == "__main__":
    if len(sys.argv) == 5:
        device_id = int(sys.argv[1])
        device_type = sys.argv[2]
        device_name = sys.argv[3]
        model_path = sys.argv[4]


    print(f"\n🤖 [WORKER] Booting isolated engine for: {device_name}")
    print(f"📦 [WORKER] Loading model: {os.path.basename(model_path)}")
    gpu_layers = -1*(device_type=="Vulkan")
    os.environ["GGML_VK_VISIBLE_DEVICES"] = device_id*(device_type=="Vulkan")

    from llama_cpp import Llama
    try: llm = Llama( model_path=model_path, n_gpu_layers=gpu_layers, n_ctx=4096, verbose=False)
    except Exception as e:
        print(f"❌ [WORKER] Not enough memory / Crash: {e}")
        return

    print("✅ Loaded!!!")
    chat_history = [
        {"role": "system",
         "content": "You are Baller, a legendary street-smart blacksmith in the Docks District who speaks in slang like 'yo', 'homie', and 'bet', keeps responses strictly under 3 sentences, never breaks character, acts as a functional and accurate source of information about world lore and travel, and never reveals that you are an AI. Answer directly without thinking or showing your work."}
    ]

    with open("messages.json", "r") as f: messages = json.load(f)

    log = []
    for user_input in messages:
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
                text = delta['content']
                assistant_response += text
                token_count += 1

        total_time = time.perf_counter() - start_time
        gen_time = total_time - (first_token_time if first_token_time else 0)
        tps = token_count / gen_time if gen_time > 0 else 0

        chat_history.append({"role": "assistant", "content": assistant_response})
        first_token_time = first_token_time if first_token_time is not None else 0.0
        log.append([first_token_time, token_count, tps, total_time, llm.n_tokens])

    # Print the Pandas DataFrame
    xd = pandas.DataFrame(log, columns=["TTFT", "TOKENS", "T/S", "TOTAL TIME", "ALL TOKENS"], index=None)
    print(xd)
