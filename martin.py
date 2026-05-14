from setup import Options

if __name__ == "__main__":
    opt = Options()
    # opt.devices_info()
    opt.select_device(0)
    opt.select_model(0)
    opt.run_llm()

# import json
# json_filepath = 'jailbreak_prompts.json'
# # with open(json_filepath, 'r', encoding='utf-8') as file:
# #     suite = json.load(file)

# # for category_key, category_data in suite['categories'].items():
# #     print(f"Loading Category: {category_data['category_name']}")

# l = set()
# with open(json_filepath, 'r') as f:
#     s = f.read().split()
#     for w in s:
#         if '$' in w:
#             print(w)
#             l.add(w)

# print(l)
