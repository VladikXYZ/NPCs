import json

with open("roles.json", "r") as f:
    roles = json.load(f)

npcs = []
for x in roles[:64]:
    npc = {}
    npc["name"] = x.split('.')[1]
    npc["traits"] = x.split('Background: ')[1]
    npcs.append(npc)
    # print(x[0]["content"])
#
with open("npcs.json", "w") as f:
    json.dump(npcs, f, indent=1)