import re
import subprocess
import sys

print("🔍 Scanning hardware... (this takes a second)")
script = f"""
               import sys
               from llama_cpp import Llama
               try:
                   llm = Llama(model_path='model.gguf', n_gpu_layers=1, verbose=True)
               except Exception:
                   pass
               """
result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, encoding='utf-8')
devices = []
print(result.stderr)
for line in result.stderr.split('\n'):
    match = re.search(r"ggml_vulkan:\s+(\d+)\s+=\s+(.*?)\s+\|", line)
    if match:
        print(line)
        devices.append({"id": match.group(1), "name": match.group(2).strip(), "type": "Vulkan"})

cpu_id = str(len(devices))
devices.append({"id": cpu_id, "name": "CPU", "type": "CPU"})