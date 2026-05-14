import sys
import subprocess
from setup import Options
import json

if __name__ == "__main__":
    opt = Options()
    opt.select_device(0)
    opt.select_model(0)
    payload_device = json.dumps(opt.selected_device)
    # payload_messages = json.dumps(messages)

    subprocess.run([sys.executable, "multi.py", payload_device, opt.model(), str(0), "dsdjajh"])