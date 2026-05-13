import time
import json
import multiprocessing
from setup import Options
from engine import run_llm_task

if __name__ == "__main__":
    # multiprocessing.set_start_method("spawn")
    opt = Options()
    with open("messages.json", "r") as f:
        messages = json.load(f)

    for device in opt.devices:
        for i in range(len(opt.models)):
            if i < 1:
                model_path = opt.models_dir + opt.models[i]
                start = time.time()
                worker_process = multiprocessing.Process(target=run_llm_task, args=(device, model_path, messages))
                worker_process.start()
                worker_process.join()
                print(f"⏱️ It took: {time.time() - start:.2f} seconds")
                time.sleep(1.0)