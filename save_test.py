import json

acc = []
for i in range(3):
    conf = {"id": i, "type": "vulkan", "name": f"gpu{i}"}
    acc.append(conf)
print(acc)
CONFIG_FILE = "npc_config.json"

with open(CONFIG_FILE, "w") as f:
    json.dump(acc, f)

with open(CONFIG_FILE, "r") as f:
    xd = json.load(f)
print(xd)