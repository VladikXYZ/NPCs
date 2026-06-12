import json
import subprocess
import sys
import time

DEVICES_FILE = "devices.json"


if __name__ == '__main__':
    with open("devices.json", "r") as f:
        devices = json.load(f)

    start = time.time()
    for i in range(len(devices)):
        subprocess.run([sys.executable, "main.py", str(i)])
        break
    print(f"It took {time.time() - start} seconds")