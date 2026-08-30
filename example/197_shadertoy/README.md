# Example 197: Vulkan Shadertoy Raymarching

This example demonstrates how to write a Shadertoy-compatible raymarching fragment shader in Common Lisp S-Expressions using the `cl-cpp-generator2` transpiler, and render it using a lightweight, native Vulkan launcher under X11/XCB on Linux.

## Project Structure

- `gen.lisp`: Common Lisp script that transpiles the raymarching shader (with `smin`, SDF sphere/box, soft shadows, and phong shading) to GLSL.
- `vulkan-shadertoy-x11/`: Minimized native Vulkan launcher for Linux (X11).
  - `launcher/shaders/shadertoy/main_image.glsl`: The transpiled GLSL output file.
  - `launcher/shaders/build_shaders.sh`: Helper script to compile GLSL shaders into SPIR-V.
  - `build_scripts/build_linux_x11/build.sh`: GCC/Clang script to build the main Vulkan launcher executable.

---

## Instructions

### Step 1: Transpile the Shader from Lisp
From this folder, load `gen.lisp` into SBCL to generate the GLSL file `main_image.glsl`:
```bash
sbcl --load gen.lisp --quit
```

### Step 2: Compile the Shaders to SPIR-V
Navigate to the shader directory and compile the GLSL code to SPIR-V using `glslangValidator` (provided by your Vulkan SDK / `dev-util/glslang` on Gentoo):
```bash
cd vulkan-shadertoy-x11/launcher/shaders/
sh build_shaders.sh
```

### Step 3: Compile and Run VK_shadertoy
Compile the launcher executable using GCC and start it:
```bash
cd ../../build_scripts/build_linux_x11/
sh build.sh
./VK_shadertoy
```

---

## Interactive Controls
While the shader launcher is running, you can use the following keys:
- `Space`: Pause/resume the timeline.
- `P`: Save a screenshot as a `.bmp` file.
- `Esc`: Close the application window.


---

## Animated Earth Globe (`gen4.lisp`)

A second shader in this example renders an animated planet Earth: real
coastlines, Blue-Marble albedo and night-time city lights baked into the
shader, an independently drifting cloud shell, atmosphere, star field, city
markers with pulsing rings and light beams, and a camera tour that SLERPs
between per-marker orientation quaternions (solved in Lisp at generation time).

```bash
sh run_earth.sh              # transpile, build, render the tour, benchmark
sh run_earth.sh --bake       # additionally re-download and re-bake the datasets
```

Rendering is headless (EGL, no X11):

```bash
sbcl --noinform --non-interactive --load gen4.lisp
g++ -O3 headless_gpu_runner.cpp -lEGL -lGL -lpng -o headless_gpu_runner
./headless_gpu_runner --res 1920 1080 --frames 2 --time 6.0 \
  --shader-dir vulkan-shadertoy-x11/launcher/shaders/earth \
  --screenshot earth.png
```

- Details, data pipeline, benchmarks and debugging tools: [doc/earth_shader.md](doc/earth_shader.md)
- Generated GLSL: `vulkan-shadertoy-x11/launcher/shaders/earth/{common,buf0,main_image}.glsl`
- Baked datasets: `tools/bake_earth_data.py` → `earth_data.lisp`
- 494 FPS at 1080p / 129 FPS at 4K on an RTX A4000
- Minimum render resolution is 256x176 (the state pass stores the map data in
  its own buffer)

- Large validation images and GIFs are intentionally ignored; regenerate them locally with `run_earth.sh` or `tools/make_animation.py`.
