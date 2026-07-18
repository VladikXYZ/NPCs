import os
import json
import time
import pandas as pd
from tqdm import tqdm
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# We use the native Hugging Face Repos instead of local GGUFs
HF_MODELS = [
    "GSAI-ML/LLaDA-8B-Instruct",
    # Note: DiffusionGemma-26B requires ~18GB of VRAM even in 4-bit,
    # so it is omitted here to prevent your RTX 3060 from crashing.
]


class HFDiffusionWrapper:
    def __init__(self):
        print("⚡ Fast Start: Running HF Diffusion Benchmark on NVIDIA RTX 3060...")
        self.device = "cuda"
        self.run_test()

    def run_test(self):
        import platform
        my_pc_name = platform.node()
        dev_name = "CUDA_NVIDIA_GeForce_RTX_3060"

        LOG_DIR = f'vlad/bench_logs/{my_pc_name}/'
        os.makedirs(LOG_DIR, exist_ok=True)
        log = []

        try:
            with open("vlad/test.json", "r") as f:
                messages = json.load(f)
            with open("data_3npcs.json") as file:
                npc = json.load(file)[2]
        except FileNotFoundError as e:
            print(f"Missing data file: {e}")
            return

        system_prompt = npc["role"] + npc["shared_system_prompt"]
        num_models = len(HF_MODELS)

        # Force 4-bit quantization on the fly
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )

        for i, model_id in enumerate(HF_MODELS):
            print(f"\nLoading {model_id} via Hugging Face...")
            try:
                # trust_remote_code=True is mandatory for diffusion architectures
                tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    quantization_config=quant_config,
                    device_map="cuda"
                )
            except Exception as e:
                print(f"Failed to load {model_id}: {e}")
                for _ in range(len(messages)):
                    log.append([-1, -1, -1, -1, -1, "FAILED"])
                continue

            for user_input in tqdm(messages, desc=f"Testing {i + 1}/{num_models} {model_id.split('/')[-1]}",
                                   unit="prompt"):
                chat = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input}
                ]

                input_ids = tokenizer.apply_chat_template(
                    chat,
                    return_tensors="pt",
                    add_generation_prompt=True
                ).to(self.device)

                start_time = time.perf_counter()

                # The generation loop
                with torch.no_grad():
                    output_ids = model.generate(
                        input_ids,
                        max_new_tokens=64,
                        do_sample=True
                    )

                total_time = time.perf_counter() - start_time
                generated_tokens = output_ids[0][input_ids.shape[1]:]
                token_count = len(generated_tokens)

                # Diffusion doesn't stream token-by-token, so TTFT equals Total Time
                tps = token_count / total_time if total_time > 0 else 0
                log.append([total_time, tps, token_count, 0, total_time, "OK"])

            # Aggressive cleanup between models
            del model
            del tokenizer
            gc.collect()
            torch.cuda.empty_cache()

        xd = pd.DataFrame(log, columns=["TTFT", "T/S", "NPC TOKENS", "USER TOKENS", "TOTAL TIME", "STATUS"])
        file_path = f"{LOG_DIR}{dev_name}_hf_diffusion.csv"
        xd.to_csv(file_path, index=False)
        print(f"\n📊 Benchmark complete! Saved to {file_path}")


if __name__ == '__main__':
    HFDiffusionWrapper()