Here is a solid **README.md** for your project. It covers the Ubuntu rig and the Windows laptop, including those specific Vulkan troubleshooting steps we just conquered.

---

# 🛠️ Baller-Engine: Cross-Platform LLM Setup

This repository contains the configuration and logic to run **llama-cpp-python** with hardware acceleration on both **Ubuntu (AMD/NVIDIA)** and **Windows (Intel Iris Xe)**.

## 🚀 Quick Start (Windows Laptop)

<a name="win-pre"></a>
### 1. Prerequisites
* **Python 3.12+**
* **Vulkan SDK:** Ensure `vulkaninfo --summary` works in your terminal or [Download here](https://vulkan.lunarg.com/sdk/home) (Tested on 1.4.341.1).
* **CMake:** Ensure `cmake --version` works in your terminal or [Download here](https://cmake.org/download/) (Tested on cmake-4.3.2-windows-x86_64.msi).

### 2. Cerate Virtual Environment

### 3. Force Vulkan Build
Try this:
```powershell
$env:CMAKE_ARGS="-DGGML_VULKAN=on"
pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir --no-binary llama-cpp-python
```

Otherwise add location of VulkanSDK:
```powershell
$env:CMAKE_ARGS="-DGGML_VULKAN=on -DVulkan_SDK='C:\VulkanSDK\1.4.341.1' -DVulkan_INCLUDE_DIR='C:\VulkanSDK\1.4.341.1\Include' -DVulkan_LIBRARY='C:\VulkanSDK\1.4.341.1\Lib\vulkan-1.lib'"
pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir --no-binary llama-cpp-python
```

---

## 🐧 Linux Setup (Ubuntu 24.04)


### 1. Prerequisites
* Jump to: [prerequsities](#win-pre)

### 2. Build Command
```bash
export CMAKE_ARGS="-DGGML_VULKAN=on"
pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir --no-binary llama-cpp-python
```

---

## 🏗️ Hardware Selection Logic
The included `main.py` features a **Zero-Footprint Fingerprinting** system:
* **Auto-Scanning:** Probes the system for Vulkan-compatible IDs.
* **Persistence:** Saves your choice to `npc_config.json`.
* **Safety:** If you move your SSD to a different machine, the script detects the hardware change and prompts for a new selection.

### Usage
```bash
python main.py
```

---

## 🦾 Model Offloading (The "Baller" Settings)
To get the best performance out of integrated graphics (like Iris Xe) or dedicated GPUs:
* **`n_gpu_layers=-1`**: Moves all model layers to the GPU.
* **`n_ctx=2048`**: Sets the context window for NPC memory.

---

## 🐞 Troubleshooting
* **UnicodeDecodeError:** Windows terminals (cp1250) often struggle with GGUF log symbols. The script handles this by forcing `utf-8` encoding during hardware probes.
* **Missing glslc:** Ensure the Vulkan `Bin` folder is in your system PATH so CMake can find the shader compiler.

---