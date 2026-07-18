import os
import json
import time
import subprocess
import pandas as pd

MODEL_DIR = 'models_diff/'
DEVICES_FILE = "devices.json"
# This path points perfectly to the binary you just compiled!
CLI_PATH = "./llama.cpp/build/bin/llama-diffusion-cli"


def get_devices():
    if os.path.exists(DEVICES_FILE):
        with open(DEVICES_FILE, "r") as f:
            return json.load(f)
    return []


def run_diffusion_benchmark():
    # Ensure the directory exists to avoid errors
    os.makedirs(MODEL_DIR, exist_ok=True)
    models = [os.path.join(MODEL_DIR, x) for x in os.listdir(MODEL_DIR) if x.endswith(".gguf")]
    devices = get_devices()

    if not devices:
        print("No devices found in devices.json!")
        return

    if not models:
        print(f"No .gguf files found in {MODEL_DIR}!")
        return

    log = []
    test_prompt = "You are a grumpy tavern keeper. Can I get a room for the night?"

    for device in devices:
        dev_name = f"{device['type']}_{device['name'].replace(' ', '_')}"
        print(f"\n🎮 Benchmarking on: {dev_name}")

        # Isolate the specific GPU via environment variables
        env = os.environ.copy()
        if device["type"] == "Vulkan":
            env["GGML_VK_VISIBLE_DEVICES"] = str(device["id"])
        elif device["type"] == "CUDA":
            env["CUDA_VISIBLE_DEVICES"] = str(device["id"])

        for model in models:
            model_info = os.path.basename(model)
            print(f"  Loading {model_info}...")

            # The diffusion CLI uses specific arguments
            cmd = [
                CLI_PATH,
                "-m", model,
                "-p", test_prompt,
                "-n", "64",  # Max tokens to generate
                "-ngl", "99",  # Offload all layers to GPU
            ]
            if "diffucoder" in model.lower():
                cmd.extend(["--diffusion-eps", "0.0001"])

            start_time = time.perf_counter()
            try:
                # Capture the output to parse the speed metrics
                result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=120)
                total_time = time.perf_counter() - start_time

                if result.returncode != 0:
                    print(f"    ❌ Crash/Error: {result.stderr.splitlines()[-1] if result.stderr else 'Unknown Error'}")
                    log.append([dev_name, model_info, -1, -1, "FAILED"])
                    continue

                # Diffusion models generate a block at once, so we estimate TPS based on the requested token count
                tps = 64 / total_time
                print(f"    ✅ Success! Total Time: {total_time:.2f}s | Est. Speed: {tps:.2f} T/s")
                log.append([dev_name, model_info, total_time, tps, "OK"])

            except subprocess.TimeoutExpired:
                print("    ⏳ Timeout! Model hung for over 120 seconds.")
                log.append([dev_name, model_info, -1, -1, "TIMEOUT"])

            # Give the GPU driver time to flush VRAM between runs
            time.sleep(5)

    # Save results
    os.makedirs("vlad/bench_logs/", exist_ok=True)
    df = pd.DataFrame(log, columns=["Device", "Model", "Total Time (s)", "Est. T/S", "Status"])
    df.to_csv("vlad/bench_logs/diffusion_results.csv", index=False)
    print("\n📊 Benchmark complete! Saved to vlad/bench_logs/diffusion_results.csv")


if __name__ == "__main__":
    run_diffusion_benchmark()