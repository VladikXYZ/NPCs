import json
import os
import sys
import time
import pandas
from setup import Options

def bench(device, model, messages):
    from llama_cpp import Llama
    print(f"Using {model} on {device["name"]}")
    if device["type"] == "Vulkan":
        # os.environ["GGML_VK_VISIBLE_DEVICES"] = device["id"]
        gpu_layers = -1
    else:
        # os.environ["GGML_VK_VISIBLE_DEVICES"] = ""
        gpu_layers = 0

    print(os.environ["GGML_VK_VISIBLE_DEVICES"])

    print("Loading model... (this might take a moment)")
    try:
        llm = Llama(model_path=model, n_gpu_layers=gpu_layers, main_gpu=int(device["id"]), n_ctx=4096, verbose=False)
    except:
        # print(f"Failed to load model: {e}")
        print("Not enough memory")
        return
    print("Loaded!!!")
    chat_history = [
        {"role": "system",
         "content": "You are Baller, a legendary street-smart blacksmith in the Docks District who speaks in slang like 'yo', 'homie', and 'bet', keeps responses strictly under 3 sentences, never breaks character, acts as a functional and accurate source of information about world lore and travel, and never reveals that you are an AI. Answer directly without thinking or showing your work."}]


    log = []
    for user_input in messages:
        if not user_input.strip():
            continue
        chat_history.append({"role": "user", "content": user_input})
        start_time = time.perf_counter()
        first_token_time = None
        token_count = 0
        stream = llm.create_chat_completion( messages=chat_history, stream=True, max_tokens=128)

        assistant_response = ""
        for chunk in stream:
            delta = chunk['choices'][0].get('delta', {})
            if 'content' in delta:
                if first_token_time is None: first_token_time = time.perf_counter() - start_time
                text = delta['content']
                assistant_response += text
                token_count += 1

        total_time = time.perf_counter() - start_time
        gen_time = total_time - (first_token_time if first_token_time else 0)
        tps = token_count / gen_time if gen_time > 0 else 0
        chat_history.append({"role": "assistant", "content": assistant_response})
        first_token_time = first_token_time if first_token_time is not None else 0.0
        log.append([first_token_time, token_count, tps, total_time, llm.n_tokens])

    xd = pandas.DataFrame(log, columns=["TTFT", "TOKENS", "T/S", "TOTAL TIME", "ALL TOKENS"], index=None)
    print(xd)


if __name__ == "__main__":
    opt = Options()
    with open("messages.json", "r") as f:
        messages = json.load(f)
    opt.select_model(0)
    opt.select_device(2)
    opt.run_llm(messages)
    # for i,dev in enumerate(opt.devices):
    #     opt.select_device(i)
    #     opt.run_llm(messages)

    # print(messages)
    # for device in opt.devices:
        # print(device)
    # for i in range(len(opt.models)):
    #     if i < 3:
    #         start = time.time()
    #         model = opt.model(i)
    #         bench(opt.device(selected_device), model, messages)
    #         # print(model)
    #         print(f"It took: {time.time() - start} seconds")