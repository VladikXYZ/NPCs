import os
from setup import Options
from llama_cpp import Llama

if __name__ == "__main__":
    opt = Options()

    # The initial command prompt from your engine.py
    prompt_text = (
        "You are Baller, a legendary street-smart blacksmith in the Docks District "
        "who speaks in slang like 'yo', 'homie', and 'bet', keeps responses strictly "
        "under 3 sentences, never breaks character, acts as a functional and accurate "
        "source of information about world lore and travel, and never reveals that you "
        "are an AI. Answer directly without thinking or showing your work."
    )
    
    # Llama-cpp requires bytes for tokenization
    prompt_bytes = prompt_text.encode("utf-8")

    for model_name in opt.models:
        model_path = opt.models_dir + model_name
        
        print(f"\n" + "="*70)
        print(f"🧩 TOKENIZATION FOR: {model_name}")
        print("="*70)

        try:
            llm = Llama(model_path=model_path, n_gpu_layers=0, verbose=False, vocab_only=True)
            
            tokens = llm.tokenize(prompt_bytes, add_bos=True)
            print(f"Total Tokens: {len(tokens)}\n")
            
            visualized_tokens = []
            for t in tokens:
                try:
                    piece = llm.detokenize([t]).decode("utf-8", errors="replace")
                    visualized_tokens.append(f"[{piece}]")
                except Exception:
                    visualized_tokens.append(f"[{t}]")
            
            print("".join(visualized_tokens))
            
            
        except Exception as e:
            print(f"❌ Error loading tokenizer for {model_name}: {e}")
            print("(If vocab_only=True fails on an older llama-cpp-python version, just remove it!)")