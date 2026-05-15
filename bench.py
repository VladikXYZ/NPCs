import json
import subprocess
import sys

with open("devices.json", "r") as f:
    devices = json.load(f)

# print(devices)
for i in range(len(devices)):
    # payload_device = json.dumps(device)
    subprocess.run([sys.executable, "vladw.py", str(i)])