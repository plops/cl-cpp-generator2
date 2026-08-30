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

## 3. Instructions to Resume After Container Restart

### Step 1: Start the Container with `--graphics`
From the host machine, start the container using:
```bash
cd /workspace/src/cl-cl-generator/example/05_dockerfile_meta/source01/examples/03_ai_env
./setup02_run.sh --gpu --graphics
```
*(Or add `--host-opt` / other flags as needed).*

### Step 2: Verify NVIDIA Hardware Acceleration
Inside the restarted container, check that Vulkan detects the NVIDIA RTX A4000 GPU:
```bash
vulkaninfo --summary
```
**Expected Output**:
`GPU0: ... NVIDIA RTX A4000 (driverVersion: 610.57.04, deviceType: DISCRETE_GPU)`

### Step 3: Run the Shadertoy Raymarching Framework on GPU
Navigate to the Shadertoy example:
```bash
cd /workspace/src/cl-cpp-generator2/example/197_shadertoy
./run_all.sh
```

For automated headless screenshot capture on the GPU:
```bash
cd /workspace/src/cl-cpp-generator2/example/197_shadertoy/vulkan-shadertoy-x11/build_scripts/build_linux_x11
xvfb-run ./VK_shadertoy --screenshot_and_close
```
