import json
import subprocess
import sys

if __name__ == '__main__':
    with open("devices.json", "r") as f:
        devices = json.load(f)

    for i in range(len(devices)):
        subprocess.run([sys.executable, "vladw.py", str(i)])
        break