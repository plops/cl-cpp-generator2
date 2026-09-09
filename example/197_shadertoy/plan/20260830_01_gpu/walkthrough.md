# Walkthrough: Hardware-Accelerated GPU Shader Rendering

This document summarizes the investigation and state of running the `cl-cpp-generator2` Shadertoy framework with NVIDIA hardware acceleration (Vulkan, OpenGL, EGL, CUDA) in this environment.

---

## 1. Investigation Summary & Current State

### A. Common Lisp to GLSL Transpilation & Software Rendering
1. **Lisp S-Expression DSL**:
   - `cl-cpp-generator2` transpiles declarative signed distance function (SDF) forms and Screen-Space Shadows (SSS) / Eye-Dome Lighting (EDL) post-processing passes into GLSL fragment shaders (`buf0.glsl`, `main_image.glsl`).
   - All 6 unit tests in `gen3.lisp` pass cleanly.
2. **SPIR-V & Launcher Build**:
   - GLSL shaders compile cleanly to SPIR-V bytecode using `glslangValidator`.
   - The native Vulkan launcher `VK_shadertoy` compiles with GCC (`-lxcb -lxcb-keysyms -lvulkan`).
3. **Headless Execution & Screenshot Validation**:
   - Executing `xvfb-run ./VK_shadertoy --screenshot_and_close` under Mesa Lavapipe CPU rasterizer generated `screenshot_0.bmp` / `screenshot_0.png`, confirming correct 3D raymarching, soft shadows, and interactive GUI slider overlays.

### B. Hardware State & Driver Capabilities
1. **Host GPU**:
   - Device: **NVIDIA RTX A4000** (16 GB VRAM, Compute Capability 8.6).
   - Host Driver Version: **610.57.04** (CUDA 13.3).
2. **CUDA / Compute**:
   - CUDA compiler (`nvcc`) and runtime successfully allocate and execute kernels on the RTX A4000 GPU directly.
3. **Vulkan / Graphics Driver Limitation in Current Container**:
   - The current container was started with default `NVIDIA_DRIVER_CAPABILITIES=compute,utility`.
   - The NVIDIA Container Toolkit only mounted compute libraries (`libcuda.so.610.57.04`), omitting the corresponding user-space graphics driver (`libGLX_nvidia.so.610.57.04`, `libnvidia-glcore.so.610.57.04`, `nvidia_icd.json`).

---

## 2. Changes Made in Container Setup

Updated `cl-cl-generator/example/05_dockerfile_meta/source01/examples/03_ai_env/setup02_run.sh` with:

1. **`--graphics` / `--gpu-graphics` Option**:
   - Sets `-e NVIDIA_DRIVER_CAPABILITIES=all` and `--gpus 'all,"capabilities=compute,utility,graphics,display,video"'` so NVIDIA Container Toolkit mounts the host-matching Vulkan and OpenGL user-space driver libraries.
   - Forwards Direct Rendering Manager render nodes (`--device /dev/dri:/dev/dri` if present on the host) for headless GPU rendering without requiring an X server.
   - Sets `--ipc=host` for MIT-SHM shared memory performance with X11 / Vulkan.

2. **`--display` / `--x11` (and Headless / SSH Support)**:
   - Forwards `$DISPLAY` into the container when present.
   - Mounts `/tmp/.X11-unix:/tmp/.X11-unix:rw` to enable local X11 and SSH X11 forwarding (`localhost:10.0` / Unix socket).
   - Mounts `$XAUTHORITY` / `~/.Xauthority` into `/root/.Xauthority:ro`.

---

## 3. Post-Restart GPU Acceleration Verification & Benchmarking

### A. Environment Verification
1. **Graphics Libraries & ICD**:
   - The restarted container successfully mounts the NVIDIA 610.57.04 graphics stack:
     - `libGLX_nvidia.so.610.57.04`, `libEGL_nvidia.so.610.57.04`, `libnvidia-glcore.so.610.57.04`, `libnvidia-rtcore.so.610.57.04`
   - Configured `/etc/vulkan/icd.d/nvidia_icd.json` and `/usr/share/glvnd/egl_vendor.d/10_nvidia.json` pointing to `libEGL_nvidia.so.0`.
2. **GPU Driver Detection**:
   - **OpenGL / EGL Core 4.6.0 NVIDIA 610.57.04** is active on the **NVIDIA RTX A4000**.
   - **Vulkan 1.4 (Driver 610.57.4.0)** detects **GPU 0: NVIDIA RTX A4000** (Physical Device Type: Discrete GPU).

---

## 4. Headless GPU Shader Runner (`headless_gpu_runner`)

A dedicated headless runner (`headless_gpu_runner.cpp`) was built to execute multi-pass Shadertoy shaders directly on the NVIDIA RTX A4000 without requiring an X11 server:

```bash
cd /workspace/src/cl-cpp-generator2/example/197_shadertoy
g++ -O3 headless_gpu_runner.cpp -lEGL -lGL -lpng -o headless_gpu_runner
```

### Supported Capabilities:
- **Multi-pass Buffer A (`buf0.glsl`) & Main Image (`main_image.glsl`) simulation**.
- **Headless EGL 4.6 PBuffer / FBO rendering on discrete GPU**.
- **High-resolution PNG screenshot capture** (`test_gpu_720p.png`, `test_gpu_1080p.png`, `test_gpu_4k.png`, `test_gpu_pointcloud_1080p.png`).
- **Precision statistical GPU benchmark** (FPS, frame times, percentiles, pixel throughput).

---

## 5. Performance Benchmark Results (NVIDIA RTX A4000)

### Raymarching 3D SDF Scene (`gen3.lisp`)

| Resolution | Dimensions | Megapixels / Frame | Average Throughput (FPS) | Mean Frame Time (ms) | 99th Percentile (ms) | Pixel Rate (MPixels/sec) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **720p HD** | 1280 x 720 | 0.92 MP | **3,038.69 FPS** | **0.329 ms** | 0.542 ms | **2,800.46 MP/s** |
| **1080p FHD** | 1920 x 1080 | 2.07 MP | **1,202.01 FPS** | **0.832 ms** | 1.259 ms | **2,492.49 MP/s** |
| **1440p QHD** | 2560 x 1440 | 3.69 MP | **586.74 FPS** | **1.704 ms** | 2.092 ms | **2,162.96 MP/s** |
| **4K UHD** | 3840 x 2160 | 8.29 MP | **413.44 FPS** | **2.419 ms** | 2.952 ms | **3,429.24 MP/s** |

### Point Cloud with Screen-Space Shadows (SSS) & Eye-Dome Lighting (EDL) (`gen2.lisp`)

| Resolution | Dimensions | Average Throughput (FPS) | Mean Frame Time (ms) | 99th Percentile (ms) |
| :--- | :--- | :--- | :--- | :--- |
| **1080p FHD** | 1920 x 1080 | **111.15 FPS** | **8.997 ms** | 9.380 ms |

---

## 6. How to Run Headless GPU Tests & Generate Screenshots

```bash
cd /workspace/src/cl-cpp-generator2/example/197_shadertoy

# 1. Transpile Lisp shader definition
sbcl --load gen3.lisp --eval '(sb-ext:exit)'

# 2. Run Headless GPU Benchmark & Save Screenshot
./headless_gpu_runner --res 1920 1080 --benchmark 300 --screenshot test_gpu_1080p.png
```

