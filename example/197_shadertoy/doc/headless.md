# Headless GPU Shader Execution & Benchmarking in Docker

This guide explains how to compile, transpile, and execute hardware-accelerated shaders (`cl-cpp-generator2` Shadertoy framework) inside a Docker container without an active X11 display server.

---

## 1. Prerequisites & Container Launch

To enable NVIDIA hardware acceleration (OpenGL / EGL / Vulkan) inside the Docker container:

### Container Startup Flags
When starting the Docker container from the host machine:
```bash
docker run -it \
  --gpus 'all,"capabilities=compute,utility,graphics,display,video"' \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  --device /dev/dri:/dev/dri \
  --ipc=host \
  -v /workspace:/workspace \
  <image_name> /bin/bash
```

*(Alternatively, run `./setup02_run.sh --gpu --graphics` if using the repository runner).*

---

## 2. Container Driver Configuration

Ensure the NVIDIA ICD manifests point to the active NVIDIA EGL driver library (`libEGL_nvidia.so.0`):

### Vulkan ICD Configuration
`/etc/vulkan/icd.d/nvidia_icd.json`:
```json
{
    "file_format_version": "1.0.0",
    "ICD": {
        "library_path": "libEGL_nvidia.so.0",
        "api_version": "1.3.277"
    }
}
```

### EGL GLVND Vendor Configuration
`/usr/share/glvnd/egl_vendor.d/10_nvidia.json`:
```json
{
    "file_format_version": "1.0.0",
    "ICD": {
        "library_path": "libEGL_nvidia.so.0"
    }
}
```

### Verification Command
```bash
eglinfo -B -p surfaceless
```
**Expected Output**:
```text
Surfaceless platform:
EGL API version: 1.5
EGL vendor string: NVIDIA
OpenGL core profile renderer: NVIDIA RTX A4000/PCIe/SSE2
OpenGL core profile version: 4.6.0 NVIDIA 610.57.04
```

---

## 3. Transpilation (Common Lisp to GLSL)

The shaders are generated from declarative Common Lisp S-Expressions.

Navigate to the project directory:
```bash
cd /workspace/src/cl-cpp-generator2/example/197_shadertoy
```

### Generate 3D Raymarching SDF Shaders (`gen3.lisp`)
```bash
sbcl --load gen3.lisp --eval '(sb-ext:exit)'
```

### Generate 3D Point Cloud with SSS & EDL Shaders (`gen2.lisp`)
```bash
sbcl --load gen2.lisp --eval '(sb-ext:exit)'
```

This writes the generated GLSL code to:
- `vulkan-shadertoy-x11/launcher/shaders/shadertoy/buf0.glsl` (State / Simulation pass)
- `vulkan-shadertoy-x11/launcher/shaders/shadertoy/main_image.glsl` (Raymarching / Render pass)

---

## 4. Building the Headless GPU Runner

Compile the C++ headless EGL runner:
```bash
cd /workspace/src/cl-cpp-generator2/example/197_shadertoy
g++ -O3 headless_gpu_runner.cpp -lEGL -lGL -lpng -o headless_gpu_runner
```

---

## 5. Running Simulations & Taking Screenshots

### A. Run Simulation & Capture 1080p Screenshot
```bash
./headless_gpu_runner --res 1920 1080 --frames 60 --screenshot screenshot_1080p.png
```

### B. Run 4K Ultra HD Rendering
```bash
./headless_gpu_runner --res 3840 2160 --frames 60 --screenshot screenshot_4k.png
```

---

## 6. Performance Benchmarking

To run hardware GPU profiling and throughput analysis:
```bash
./headless_gpu_runner --res 1920 1080 --benchmark 500
```

### Measured Performance on NVIDIA RTX A4000 (16 GB VRAM, Driver 610.57.04)

| Resolution | Dimensions | Megapixels / Frame | Average FPS | Mean Frame Time | 99th Percentile | Pixel Throughput |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **720p HD** | 1280 x 720 | 0.92 MP | **3,038.69 FPS** | **0.329 ms** | 0.542 ms | **2.80 GPixels/sec** |
| **1080p FHD** | 1920 x 1080 | 2.07 MP | **1,202.01 FPS** | **0.832 ms** | 1.259 ms | **2.49 GPixels/sec** |
| **1440p QHD** | 2560 x 1440 | 3.69 MP | **586.74 FPS** | **1.704 ms** | 2.092 ms | **2.16 GPixels/sec** |
| **4K UHD** | 3840 x 2160 | 8.29 MP | **413.44 FPS** | **2.419 ms** | 2.952 ms | **3.43 GPixels/sec** |
| **Point Cloud (1080p)** | 1920 x 1080 | 2.07 MP | **111.15 FPS** | **8.997 ms** | 9.380 ms | **0.23 GPixels/sec** |

---

## 7. Command Reference

| Option | Argument | Description | Default |
| :--- | :--- | :--- | :--- |
| `--res` | `<W> <H>` | Render viewport resolution | `1280 720` |
| `--frames` | `<N>` | Number of simulation frames to step | `60` |
| `--screenshot` | `<file.png>` | File path to write RGBA PNG screenshot | `gpu_screenshot.png` |
| `--benchmark` | `[N]` | Run warmup + N-frame statistical benchmark | `500` |
| `--shader-dir` | `<path>` | Custom directory containing `buf0.glsl` & `main_image.glsl` | `vulkan-shadertoy-x11/launcher/shaders/shadertoy` |
