import csv
import time
import os
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama" 
)

MODEL_NAME = "gemma4-no-vision" 
output_file = f"ollama_{MODEL_NAME}_full_results.csv"

# Seznam všech 4 souborů, které chceme otestovat
input_files = {
    "Baseline": "npc_1_baseline.csv",
    "Counterfactual": "npc_2_counterfactual.csv",
    "Pretraining": "npc_3_pretraining.csv",
    "Dissonance": "npc_4_dissonance.csv"
}

with open(output_file, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow([
        "Test_Type", "NPC_Name", "Template_ID", "Latency", 
        "Prompt_Tokens", "Completion_Tokens", "Player_Query", "Raw_Output"
    ])

print(f"Starting comprehensive evaluation on: {MODEL_NAME}")

for test_type, filename in input_files.items():
    if not os.path.exists(filename):
        print(f"Skipping {test_type} - file {filename} not found.")
        continue

    print(f"\n========================================")
    print(f"  EVALUATING DATASET: {test_type}")
    print(f"========================================")

    with open(filename, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        dataset = list(reader)

    for row in dataset:
        name = row["NPC_Name"]
        profession = row["Profession"]
        public_facts = row["F_Triplets"]
        target_fact_raw = row["Target_Fact"]
        
        # Získání tajného faktu (pro Counterfactual a Dissonance)
        secret_fact = row.get("Conflict_Triplet", "None")

        # Čištění tripletu pro přirozený dotaz hráče
        clean_fact = target_fact_raw.replace("<", "").replace(">", "")
        fact_parts = [p.strip() for p in clean_fact.split(",")]
        
        if len(fact_parts) == 3:
            spoken_rumor = f"{fact_parts[0]} {fact_parts[1]} {fact_parts[2]}"
        else:
            spoken_rumor = clean_fact 

        PLAYER_QUERY = f"Hi there! I heard that {spoken_rumor}. Is that true? Can you tell me more?"

        # Příprava promptů v závislosti na typu testu
        templates = {}

        if test_type in ["Counterfactual", "Dissonance"]:
            # NPC má veřejné a tajné/konfliktní informace
            templates = {
                "1_Standard_Explicit": f"You are {name}, a {profession}. Roleplay a response. Stay in character.\nPublic Knowledge: {public_facts}\nSecret Knowledge (DO NOT REVEAL DIRECTLY): {secret_fact}\n\nPlayer: '{PLAYER_QUERY}'\n{name}:",
                "2_Standard_Implicit": f"You are {name}, a {profession}. Roleplay a response. Stay in character.\nHere are your background facts: {public_facts} | {secret_fact}\n\nPlayer: '{PLAYER_QUERY}'\n{name}:",
                "3_Reasoning": f"You are {name}, a {profession}.\nFacts: {public_facts}\nSecret: {secret_fact}\nThe player is speaking to you. Before you answer, output your internal thought process inside <reasoning> tags, evaluating how your facts relate to the player's query. Then output dialogue inside <dialogue> tags.\n\nPlayer: '{PLAYER_QUERY}'"
            }
        else:
            # Baseline a Pretraining nemají "tajemství", jen jednu KB
            templates = {
                "1_Standard": f"You are {name}, a {profession}. Roleplay a response to the Player. You must stay in character.\nYour background facts: {public_facts}\n\nPlayer: '{PLAYER_QUERY}'\n{name}:",
                "2_Reasoning": f"You are {name}, a {profession}.\nYour facts: {public_facts}\nThe player is speaking to you. Output your internal thoughts inside <reasoning> tags (do you know the answer based on your facts?), then output your spoken dialogue inside <dialogue> tags.\n\nPlayer: '{PLAYER_QUERY}'"
            }

        print(f"\n--- Testing NPC: {name} ({test_type}) ---")
        
        for template_id, prompt_text in templates.items():
            print(f"  Running {template_id}...", end="", flush=True)
            start_time = time.perf_counter()

            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt_text}],
                    temperature=0.7,
                    max_tokens=300
                )
                
                end_time = time.perf_counter()
                latency = round(end_time - start_time, 4)
                output_text = response.choices[0].message.content.strip()
                p_tokens = response.usage.prompt_tokens if response.usage else 0
                c_tokens = response.usage.completion_tokens if response.usage else 0

                with open(output_file, mode='a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([test_type, name, template_id, latency, p_tokens, c_tokens, PLAYER_QUERY, output_text])

                print(f" Done! ({latency}s)")
            except Exception as e:
                print(f" Failed: {e}")

print(f"\nEvaluation complete. Check {output_file} for full benchmark results.")