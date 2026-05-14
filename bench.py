import time
import json
import subprocess
import sys
from setup import Options

if __name__ == "__main__":
    opt = Options()

    with open("messages.json", "r") as f:
        messages = json.load(f)

    for device in opt.devices:
        for i in range(len(opt.models)):
            #if i < 1:
                model_path = opt.models_dir + opt.models[i]
                start = time.time()

                print(f"\n==================================================")
                print(f"🚀 DISPATCHING: {device['name']} | Model: {opt.models[i]}")
                print(f"==================================================")

                # 1. Freeze-dry your Python objects into JSON strings
                payload_device = json.dumps(device)
                payload_messages = json.dumps(messages)

                # 2. Fire the subprocess (Acts exactly like typing in the terminal)
                subprocess.run([sys.executable, "engine.py", payload_device, model_path, payload_messages])

                print(f"⏱️ It took: {time.time() - start:.2f} seconds")
                time.sleep(1.0)