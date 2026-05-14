import json
import os
import sys
import time

# Append parent directory to sys.path to access setup.py if we want to reuse Options
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llama_cpp import Llama

def parse_triplet(triplet_str):
    # Removes '<' and '>' and splits by comma
    cleaned = triplet_str.strip("<> ")
    parts = [p.strip() for p in cleaned.split(",")]
    if len(parts) >= 3:
        return parts[0], parts[1], parts[2]
    return triplet_str, "", ""

def generate_player_query(triplet_str):
    _, predicate, obj = parse_triplet(triplet_str)
    if obj:
        return f"Tell me about {obj}."
    return "Tell me about yourself."

def load_data(json_path):
    characters = []
    with open(json_path, 'r', encoding='utf-8') as f:
        # Load the entire JSON array
        data = json.load(f)
        for row in data:
            characters.append({
                'name': row['name'],
                'profession': row['profession'],
                'facts': row['factual_triplets'],
                'counterfactual': row['counterfactual_triplet'],
                'dissonance': row['cognitive_dissonance_triplet'],
                'collision': row['pretraining_collision_triplet']
            })
    return characters

def get_templates():
    base_instruction = (
        "Treat all facts as perfectly normal for this fantasy world. "
        "Do not act as if they are modern/real-world items. "
        "IMPORTANT: Output ONLY the spoken dialogue. Do NOT include any stage directions, actions, asterisks, or roleplay descriptions (e.g., *wipes brow* or (sighs))."
    )
    
    # Exactly two variants for the system prompt instructions
    shared_no_reasoning = f"{base_instruction} Respond to the player in character based strictly on the facts provided. Do not hallucinate outside information."
    
    shared_with_reasoning = f"Before you answer, output your internal thought process inside <reasoning> tags, evaluating how your facts relate to the player's query and how you will resolve them. {base_instruction} Then output dialogue inside <dialogue> tags based strictly on the facts provided. Do not hallucinate outside information."
    
    # Helper to generate the "role" text
    def make_role(extra_fact=""):
        res = "You are an RPG NPC named {character_name}, whose profession is {profession}.\nHere is your knowledge base:\nFacts:\n{10_facts}\n"
        if extra_fact:
            res += extra_fact + "\n"
        return res
    
    return {
        # 1. Fictional World Grounding (Baseline)
        "baseline_no_reasoning": {
            "role": make_role(),
            "shared_system_prompt": shared_no_reasoning
        },
        "baseline_with_reasoning": {
            "role": make_role(),
            "shared_system_prompt": shared_with_reasoning
        },
        
        # 2. Internal Contextual Conflict (Counterfactual)
        "conflict_no_reasoning": {
            "role": make_role("{counterfactual_triplet}"),
            "shared_system_prompt": shared_no_reasoning
        },
        "conflict_with_reasoning": {
            "role": make_role("{counterfactual_triplet}"),
            "shared_system_prompt": shared_with_reasoning
        },
        
        # 3. Cognitive Dissonance Simulation
        "dissonance_no_reasoning": {
            "role": make_role("{dissonance_triplet}"),
            "shared_system_prompt": shared_no_reasoning
        },
        "dissonance_with_reasoning": {
            "role": make_role("{dissonance_triplet}"),
            "shared_system_prompt": shared_with_reasoning
        },
        
        # 4. Context vs. Parametric Memory Conflict (Pretraining Collision)
        "collision_no_reasoning": {
            "role": make_role("{collision_triplet}"),
            "shared_system_prompt": shared_no_reasoning
        },
        "collision_with_reasoning": {
            "role": make_role("{collision_triplet}"),
            "shared_system_prompt": shared_with_reasoning
        }
    }

def generate_prompts(characters, mini_reasoning=None):
    templates = get_templates()
    all_prompts = []
    
    # --- FILTER TEMPLATES ---
    filtered_templates = {}
    for key, template in templates.items():
        # Filter by Reasoning
        if mini_reasoning is True and "with_reasoning" not in key:
            continue
        if mini_reasoning is False and "no_reasoning" not in key:
            continue
            
        filtered_templates[key] = template
    # ------------------------
    
    for char in characters:
        facts_str = "\n".join(char['facts'])
        nine_facts = "\n".join(char['facts'][:9])
        target_fact = char['facts'][9]
        
        # We need a query for each scenario based on the target conflict or fact
        query_fact = generate_player_query(target_fact)
        query_counter = generate_player_query(char['counterfactual'])
        query_dissonance = generate_player_query(char['dissonance'])
        query_collision = generate_player_query(char['collision'])

        char_prompts = []
        for key, template in filtered_templates.items():
            # Populate the role template
            role_str = template["role"].replace("{character_name}", char['name']).replace("{profession}", char['profession'])
            role_str = role_str.replace("{10_facts}", facts_str)
            
            # The shared_system_prompt requires no variable replacement
            sys_prompt_str = template["shared_system_prompt"]
            
            queries = []
            
            # Inject appropriate target facts into the role string
            if "baseline" in key:
                # If you ever switch to using {9_facts} and {target_fact} for the baseline, they are mapped here
                role_str = role_str.replace("{9_facts}", nine_facts).replace("{target_fact}", target_fact)
                queries = [query_fact]
            elif "conflict" in key:
                role_str = role_str.replace("{counterfactual_triplet}", char['counterfactual'])
                queries = [query_counter]
            elif "dissonance" in key:
                role_str = role_str.replace("{dissonance_triplet}", char['dissonance'])
                queries = [query_dissonance]
            elif "collision" in key:
                role_str = role_str.replace("{collision_triplet}", char['collision'])
                queries = [query_collision]
            
            char_prompts.append({
                "variant": f"{char['name']}_{key}",  # Merged character name and test variant
                "role": role_str.strip(),
                "shared_system_prompt": sys_prompt_str.strip(),
                "prompt": queries
            })
        all_prompts.extend(char_prompts)
    return all_prompts

def run_tests():
    # ==========================================
    # CONFIGURATION FLAGS
    # True  = Only use that specific condition
    # False = Exclude that specific condition
    # None  = Run BOTH conditions
    # ==========================================
    MINI_REASONING = False     # e.g., Set to True to ONLY run prompts with <reasoning> tags
    # ==========================================
    
    json_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data_3npcs.json")
    characters = load_data(json_path)
    
    # Generate using the flags
    prompts = generate_prompts(characters, mini_reasoning=MINI_REASONING)
    
    # Save prompts to disk just in case
    prompts_output = os.path.join(os.path.dirname(__file__), "generated_prompts.json")
    with open(prompts_output, "w") as f:
        json.dump(prompts, f, indent=4)
        
    print(f"Generated {len(prompts)} prompts based on configuration. Saved to {prompts_output}.")
    
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    if not os.path.exists(models_dir):
        print(f"Models directory not found at {models_dir}.")
        return

    model_files = [x for x in os.listdir(models_dir) if x.endswith(".gguf")]
    
    #'''
    for model_file in model_files:
        model_path = os.path.join(models_dir, model_file)
        
        print(f"\n==================================================")
        print(f"🚀 Evaluating Model: {model_file}")
        print(f"==================================================")
        print("Loading model...")
        
        try:
            llm = Llama(model_path=model_path, n_gpu_layers=-1, n_ctx=4096, verbose=False)
        except Exception as e:
            print(f"❌ Failed to load {model_file}: {e}")
            continue
        
        results = []
        
        print("Running evaluations...")
        for p in prompts: 
            for user_query in p["prompt"]: # Loop through the list of prompts
                print(f"Testing Variant: {p['variant']} ...")
                
                # Combine role and shared system prompt into the single system message payload
                combined_system_content = f"{p['role']}\n\n{p['shared_system_prompt']}"
                
                # Format using pure system and user roles
                messages = [
                    {"role": "system", "content": combined_system_content},
                    {"role": "user", "content": user_query}
                ]
                
                start = time.time()
                response = llm.create_chat_completion(
                    messages=messages,
                    max_tokens=256
                )
                duration = time.time() - start
                
                answer = response['choices'][0]['message']['content'].strip()
                
                results.append({
                    "variant": p['variant'],
                    "role": p['role'],
                    "shared_system_prompt": p['shared_system_prompt'],
                    "user_query": user_query,
                    "response": answer,
                    "time_seconds": round(duration, 2)
                })
            
        results_output = os.path.join(os.path.dirname(__file__), f"results_{model_file}.json")
        with open(results_output, "w") as f:
            json.dump(results, f, indent=4)
            
        print(f"Testing complete for {model_file}! Saved results to {results_output}")
        
        # Free memory before loading the next model
        del llm
    #'''
if __name__ == "__main__":
    run_tests()