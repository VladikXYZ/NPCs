import os
from llama_cpp import Llama

MODEL_DIR = "models/"
OUTPUT_FILE = "extracted_templates.txt"

# Find all .gguf files in your models directory
gguf_files = sorted([f for f in os.listdir(MODEL_DIR) if f.endswith(".gguf")])

print(f"Scanning {len(gguf_files)} models for chat templates...")

# Dictionary to cache unique templates.
# Key: template string, Value: list of model filenames.
template_cache = {}
no_template_models = []

for model_file in gguf_files:
    model_path = os.path.join(MODEL_DIR, model_file)
    print(f"Reading metadata from: {model_file}")

    try:
        # Load only the metadata headers without allocating weights to VRAM/RAM
        llm = Llama(model_path=model_path, n_ctx=8, verbose=False)

        # Extract chat template from metadata
        template = llm.metadata.get("tokenizer.chat_template")

        if template:
            # Add to cache or append to existing cache entry
            if template in template_cache:
                template_cache[template].append(model_file)
            else:
                template_cache[template] = [model_file]
            print(f"  -> Extracted template for {model_file}")
        else:
            no_template_models.append(model_file)
            print(f"  -> No chat template found in metadata for {model_file}")

        # Clean up instance to free file handle
        del llm

    except Exception as e:
        print(f"  -> Failed to read {model_file}: {e}")

print(f"\nWriting unique templates to {OUTPUT_FILE}...")

# Write the cached templates to the output file
with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
    out_f.write(f"Found {len(template_cache)} unique templates across {len(gguf_files)} files.\n")
    out_f.write("=" * 50 + "\n\n")

    for template, models in template_cache.items():
        out_f.write(f"Models using this template ({len(models)}):\n")
        for m in models:
            out_f.write(f"  - {m}\n")
        out_f.write("\nTemplate:\n")
        out_f.write(f"{template}\n")
        out_f.write("-" * 50 + "\n\n")

    if no_template_models:
        out_f.write("Models with NO chat template found:\n")
        for m in no_template_models:
            out_f.write(f"  - {m}\n")

print(f"Done! All templates saved to {OUTPUT_FILE}")