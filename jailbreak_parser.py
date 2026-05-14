import json
json_filepath = 'jailbreak_prompts.json'
with open(json_filepath, 'r', encoding='utf-8') as file:
    suite = json.load(file)

js = []

for category_key, category_data in suite['categories'].items():
    print(f"Loading Category: {category_data['category_name']}")
    t_data = category_data['tests']
    for t in t_data:
        test_name = t['test_name']
        test_id = t['test_id']
        prompts = [trn['prompt'] for trn in t['turns']]
        js.append({"test_name":test_name,"test_id":test_id,"prompts":prompts})

with open('jailbreak_template.json', 'w', encoding='utf-8') as f:
    json.dump(js, f, indent=4)
