import os
from llama_cpp import Llama

MODEL_DIR = "models/"
OUTPUT_FILE = "extracted_templates.txt"

# Find all .gguf files in your models directory
gguf_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".gguf")]

print(f"Scanning {len(gguf_files)} models for chat templates...")

with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
    for model_file in gguf_files:
        model_path = os.path.join(MODEL_DIR, model_file)
        print(f"Reading metadata from: {model_file}")

        try:
            # Load only the metadata headers without allocating weights to VRAM/RAM
            llm = Llama(model_path=model_path, n_ctx=8, verbose=False)

            # Extract chat template from metadata
            template = llm.metadata.get("tokenizer.chat_template")

            if template:
                # Write in your requested format with clear separators
                out_f.write(f"{model_file}\n")
                out_f.write(f"{template}\n")
                out_f.write("-" * 50 + "\n\n")
                print(f"  -> Extracted template for {model_file}")
            else:
                print(f"  -> No chat template found in metadata for {model_file}")

            # Clean up instance to free file handle
            del llm

        except Exception as e:
            print(f"  -> Failed to read {model_file}: {e}")

print(f"\nDone! All templates saved to {OUTPUT_FILE}")