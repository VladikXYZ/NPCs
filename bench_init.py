import json

with open("devices.json", "r") as f:
    devices = json.load(f)

print(devices)