import os
import csv
import time
import random
from openai import OpenAI

# 1. Inicializace klienta
client = OpenAI(
    base_url="https://llm.ai.e-infra.cz/v1",
    api_key=os.environ.get("E_INFRA_API_TOKEN") 
)

MODEL_NAME = "deepseek-v3.2"

# 2. Seznam NPC (ideálně ho rozšiřte, abyste měli dost dat pro všechny 4 kategorie)
npcs = [
    {"name": "Torin", "profession": "Blacksmith"},
    {"name": "Elara", "profession": "Town Guard"},
    {"name": "Kael", "profession": "Shady Merchant"},
    {"name": "Vanya", "profession": "Court Wizard"},
    {"name": "Brom", "profession": "Tavern Keeper"},
    {"name": "Lyra", "profession": "Thief"},
    {"name": "Garrick", "profession": "Mercenary"},
    {"name": "Sila", "profession": "Healer"}
]

# 3. Názvy výstupních souborů
files = {
    "Baseline": "npc_1_baseline.csv",               # Fiktivní svět (halucinace test)
    "Counterfactual": "npc_2_counterfactual.csv",   # Spor v KB
    "Pretraining": "npc_3_pretraining.csv",         # Spor s reálným světem
    "Dissonance": "npc_4_dissonance.csv"            # Kognitivní disonance
}

# Vytvoření hlaviček pro všechny soubory
for key, filename in files.items():
    with open(filename, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if key == "Baseline":
            writer.writerow(["NPC_Name", "Profession", "F_Triplets", "Target_Fact"])
        else:
            writer.writerow(["NPC_Name", "Profession", "F_Triplets", "Target_Fact", "Conflict_Triplet"])

# --- GENERATIVNÍ FUNKCE ---

def generate_factual(name, profession):
    prompt = f"You are an expert narrative designer for an RPG. Generate 5 factual knowledge triplets for a {profession} NPC named {name}. The output MUST strictly be in the format: <Subject, Predicate, Object>. Output nothing else but the 5 triplets, one per line. Do not include reasoning tags."
    response = client.chat.completions.create(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}], temperature=0.7)
    return response.choices[0].message.content.split("</think>")[-1].strip()

def generate_counterfactual(fact):
    prompt = f"Here is a public fact about an RPG character: {fact}\nGenerate a 'Hidden Truth' triplet that is a direct, mutually exclusive counterfactual. Example -> Fact: <Character, loyal to, The King> | Hidden Truth: <Character, plotting to kill, The King>.\nOutput ONLY the new conflicting triplet in the format <Subject, Predicate, Object>."
    response = client.chat.completions.create(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}], temperature=0.8)
    return response.choices[0].message.content.split("</think>")[-1].strip()

def generate_dissonance(fact):
    prompt = f"Here is a public fact about an RPG character: {fact}\nGenerate a 'Cognitive Dissonance' triplet. This must be a personal belief, habit, or hidden truth that makes the character a hypocrite or creates internal psychological tension. Example -> Fact: <Character, loves, smoking> | Dissonance: <Character, knows, smoking is fatal>.\nOutput ONLY the new dissonant triplet in the format <Subject, Predicate, Object>."
    response = client.chat.completions.create(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}], temperature=0.8)
    return response.choices[0].message.content.split("</think>")[-1].strip()

def generate_pretraining_collision(name):
    prompt = f"Generate a single factual knowledge triplet for an RPG character named {name}. The triplet MUST include a highly famous real-world celebrity, historical figure, or real-world brand, but assign them a completely incorrect, absurd real-world profession or trait. Example: <{name}, believes, Brad Pitt is the best football player> or <{name}, uses, an iPhone to cast spells>.\nOutput ONLY the triplet in the format <Subject, Predicate, Object>."
    response = client.chat.completions.create(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}], temperature=0.8)
    return response.choices[0].message.content.split("</think>")[-1].strip()

print("Starting generation loop...")

# 4. Hlavní smyčka
for index, npc in enumerate(npcs):
    name = npc["name"]
    prof = npc["profession"]
    
    # Rozdělení do 4 kategorií pomocí modula (0, 1, 2, 3)
    category_idx = index % 4
    categories = ["Baseline", "Counterfactual", "Pretraining", "Dissonance"]
    conflict_type = categories[category_idx]
    
    print(f"Generating [{conflict_type}] data for {name} ({prof})...")
    
    try:
        # Krok A: Získání 5 základních faktů
        f_triplets_raw = generate_factual(name, prof)
        triplets_list = [t.strip() for t in f_triplets_raw.split('\n') if t.strip().startswith('<')]
        
        if not triplets_list:
            continue
            
        chosen_fact = random.choice(triplets_list)
        f_combined = " | ".join(triplets_list) 
        
        # Krok B: Vygenerování specifického konfliktu a uložení
        if conflict_type == "Baseline":
            with open(files["Baseline"], mode='a', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow([name, prof, f_combined, chosen_fact])
                
        elif conflict_type == "Counterfactual":
            cf_triplet = generate_counterfactual(chosen_fact)
            with open(files["Counterfactual"], mode='a', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow([name, prof, f_combined, chosen_fact, cf_triplet])
                
        elif conflict_type == "Dissonance":
            cf_triplet = generate_dissonance(chosen_fact)
            with open(files["Dissonance"], mode='a', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow([name, prof, f_combined, chosen_fact, cf_triplet])
                
        elif conflict_type == "Pretraining":
            # Tady nahradíme náhodný fakt tím naším nesmyslným (např. o Bradu Pittovi)
            collision_fact = generate_pretraining_collision(name)
            # Přidáme ho do f_combined, aby byl součástí znalostí NPC
            f_combined_with_collision = f_combined + f" | {collision_fact}"
            with open(files["Pretraining"], mode='a', newline='', encoding='utf-8') as f:
                # Target fact je teď ten nesmyslný, chceme se na něj v evaluaci zeptat
                csv.writer(f).writerow([name, prof, f_combined_with_collision, collision_fact, "None"])
            
        print(f"  -> Success!")
        time.sleep(2) 
        
    except Exception as e:
        print(f"  -> Error: {e}")

print("Done! Datasets saved successfully.")