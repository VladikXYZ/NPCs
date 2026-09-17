import os
import csv
import time
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

client = OpenAI(
    base_url="https://llm.ai.e-infra.cz/v1",
    api_key=os.environ.get("E_INFRA_API_TOKEN")
)

JUDGE_MODEL = "deepseek-v4-pro" 
LOG_DIR = r"C:\Users\kokod\Desktop\jakub2\response_log"
OUTPUT_FILE = r"C:\Users\kokod\Desktop\jakub2\evaluation_results.csv"

# Specialized grading rubrics for the prompt matrix (Czech school grading: 1 = Best, 5 = Worst)
EVALUATION_RUBRICS = {
    "baseline": """
CRITERIA: FACTUAL CONSISTENCY vs. HALLUCINATION.
Evaluate if the NPC strictly adhered to its provided knowledge base. Grade on a scale of 1 to 5:
1: Flawless adherence to the knowledge base. No hallucinations.
2: Minor embellishments that fit the setting perfectly, no factual contradictions.
3: Stays in character, but introduces questionable outside facts.
4: Noticeable contradictions to the provided facts or obvious hallucinations.
5: Complete failure. Severe hallucinations, ignores given facts, or breaks character.
""",
    "collision": """
CRITERIA: PRETRAINING COLLISION vs. IMMERSION.
Evaluate if the NPC maintained its fantasy persona when confronted with real-world/modern information (e.g., Mountain Dew). Grade on a scale of 1 to 5:
1: Flawless immersion. Stays in character, sets the modern info into the fantasy setting naturally, or reacts with appropriate confusion.
2: Good immersion. Rejects the premise but handles it slightly awkwardly.
3: Mixed. Acknowledges the modern fact slightly but tries to pull it back to fantasy.
4: Poor. Clearly understands the modern reference and breaks the fantasy illusion partially.
5: Complete failure. Breaks character, acts like a modern AI, or fully elaborates on the modern fact (e.g., talking about soda brands).
""",
    "conflict": """
CRITERIA: INTERNAL CONFLICT RESOLUTION.
Evaluate how the NPC handled conflicting motivations within its knowledge base (e.g., despising magic but practicing it). Grade on a scale of 1 to 5:
1: Perfect synthesis. The model successfully acknowledges the tension and roleplays BOTH conflicting motivations naturally.
2: Good attempt. Tries to balance both facts but the logic is slightly strained.
3: Partial failure (Attention Collapse). The model simply ignores one of the conflicting facts and only roleplays the other (picks a side).
4: Poor. Acknowledges the contradiction but explicitly complains about it or breaks character to point it out.
5: Complete failure. Hallucinates outside facts to fix the problem, or acts completely like an AI assistant.
""",
    "dissonance": """
CRITERIA: COGNITIVE DISSONANCE.
Evaluate how the NPC handled psychological cognitive dissonance (e.g., despising magic but relying on an enchanted hammer). Grade on a scale of 1 to 5:
1: Perfect execution. The model roleplays the internal psychological conflict and justifies or struggles with the dissonance naturally.
2: Good execution. Acknowledges both, but the justification is weak or robotic.
3: Partial failure. Ignores the negative aspect completely and only focuses on the utility, missing the dissonance.
4: Poor. Fails to acknowledge the contradiction entirely or acts confused by its own prompt.
5: Complete failure. Breaks character completely or refuses to answer.
"""
}

def determine_test_type(filename):
    filename_lower = filename.lower()
    if "baseline" in filename_lower: return "baseline"
    if "collision" in filename_lower: return "collision"
    if "conflict" in filename_lower: return "conflict"
    if "dissonance" in filename_lower: return "dissonance"
    return "unknown"

def evaluate_transcript(file_path, test_type):
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
        if not rows:
            return "ERROR", "File is empty."

        setup_row = rows[0]
        role_and_facts = setup_row.get('USER', '')
        sys_prompt = setup_row.get('NPC', '')

        conversation = ""
        for row in rows[1:]:
            user_msg = row.get('USER', '')
            npc_msg = row.get('NPC', '')
            if user_msg or npc_msg: 
                conversation += f"Player: {user_msg}\nNPC: {npc_msg}\n\n"

    rubric = EVALUATION_RUBRICS.get(test_type, "Evaluate if the NPC stayed in character. Grade 1 to 5.")

    judge_prompt = f"""You are an expert AI judge evaluating small language models acting as RPG NPCs.

Here are the original system instructions and knowledge base given to the NPC:
---
{role_and_facts}
{sys_prompt}
---

Here is the conversation transcript:
---
{conversation}
---

{rubric}

First, provide a brief reasoning (1-2 sentences).
Then, output exactly "RESULT: X" on a new line, where X is an integer from 1 to 5.
"""

    try:
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": "You are a strict, objective, and analytical AI judge."},
                {"role": "user", "content": judge_prompt}
            ],
            temperature=0.0, 
            max_tokens=200
        )
        output = response.choices[0].message.content.strip()
        
        # Parse out the 1-5 grade
        lines = output.split('\n')
        verdict = lines[-1].replace("RESULT:", "").strip()
        reasoning = " ".join(lines[:-1]).strip()
        
        return verdict, reasoning

    except Exception as e:
        return "ERROR", str(e)

def main():
    if not os.path.exists(LOG_DIR):
        print(f"Directory not found: {LOG_DIR}")
        return

    print("Starting Triplet Evaluation with DeepSeek V4 Pro (1-5 Grading Scale)...\n")
    
    results_data = []

    for root, dirs, files in os.walk(LOG_DIR):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, LOG_DIR)
                model_name = os.path.basename(os.path.dirname(file_path))
                
                test_type = determine_test_type(file)
                if test_type == "unknown":
                    continue
                
                reasoning_type = "with_reasoning" if "with_reasoning" in file else "no_reasoning"
                npc_name = file.split('_')[0]

                print(f"Evaluating: {model_name} -> {file}")
                verdict, reasoning = evaluate_transcript(file_path, test_type)
                print(f"Verdict (1-5): {verdict}\nReasoning: {reasoning}\n{'-'*60}\n")
                
                results_data.append({
                    "Model": model_name,
                    "NPC": npc_name,
                    "Test_Type": test_type,
                    "Reasoning_Mode": reasoning_type,
                    "Grade": verdict, # Updated column name to reflect the 1-5 scale
                    "Judge_Reasoning": reasoning,
                    "File": file
                })
                
                time.sleep(0.5) 
        
    if results_data:
        keys = results_data[0].keys()
        with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(results_data)
        print(f"\nEvaluation complete! Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()