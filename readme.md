# 🛠️ Baller-Engine: Cross-Platform LLM Setup

This repository contains the configuration and logic to run **llama-cpp-python** with hardware acceleration on both **Ubuntu (AMD/NVIDIA)** and **Windows (Intel Iris Xe)**.

## 🚀 Quick Start

### Prerequisites
* **Python 3.12+**
* **Vulkan SDK:** Ensure `vulkaninfo --summary` works in your terminal or [Download here](https://vulkan.lunarg.com/sdk/home) (Tested on 1.4.341.1).
* **CMake:** Ensure `cmake --version` works in your terminal or [Download here](https://cmake.org/download/) (Tested on cmake-4.3.2-windows-x86_64.msi).
* **venv:** Create virtual environment

### Windows 11
Install llama-cpp-python with this command:
```powershell
$env:CMAKE_ARGS="-DGGML_VULKAN=on"; pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir --no-binary llama-cpp-python
```
<!--
Otherwise add location of VulkanSDK:
```powershell
$env:CMAKE_ARGS="-DGGML_VULKAN=on -DVulkan_SDK='C:\VulkanSDK\1.4.341.1' -DVulkan_INCLUDE_DIR='C:\VulkanSDK\1.4.341.1\Include' -DVulkan_LIBRARY='C:\VulkanSDK\1.4.341.1\Lib\vulkan-1.lib'"; pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir --no-binary llama-cpp-python
```
-->
---

### 🐧 Linux Setup (Ubuntu 24.04)
Install llama-cpp-python with this command:
```bash
export CMAKE_ARGS="-DGGML_VULKAN=on"
pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir --no-binary llama-cpp-python
```

---

## 🏗️ Hardware Selection Logic
The included `main.py` features a **Zero-Footprint Fingerprinting** system:
* **Auto-Scanning:** Probes the system for Vulkan-compatible IDs.
* **Persistence:** Saves your choice to `devices.json`.
* **Safety:** If you move your SSD to a different machine, the script detects the hardware change and prompts for a new selection.

### Usage
```bash
python main.py
```

---
