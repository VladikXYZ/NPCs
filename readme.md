# Cross-Platform LLM Setup

This repository contains the configuration and logic to run **llama-cpp-python** with hardware acceleration on all Vulkan supported devices.

## 🚀 Quick Start

### Prerequisites

*   **Python 3.12+**
    *   Ensure Python is installed and added to your system PATH.
*   **Vulkan SDK** (Tested on 1.4.341.1)
    *   [Download here](https://vulkan.lunarg.com/sdk/home) if not installed. Verify your installation with:
    ```bash
    vulkaninfo --summary
    ```
    * Vulkan SDK required components:
      * The Vulkan SDK Core
      * GLM Headers.
      * SDL libraries and headers.
      * Volk header, source, and library.
      * Vulkan Memory Allocator header.
*   **CMake** (Tested on 4.3.2)
    *   [Download here](https://cmake.org/download/) if missing. Verify with:
    ```bash
    cmake --version
    ```
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

### 🐧 Linux Setup (Ubuntu 24.04 and CachyOS)
Install llama-cpp-python with this command:
```bash
export CMAKE_ARGS="-DGGML_VULKAN=on"
pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir --no-binary llama-cpp-python
```
Install llama-cpp-python using ninja with this command:
```bash
CMAKE_ARGS="-DGGML_VULKAN=on -GNinja" pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir --no-binary llama-cpp-python
```

---
