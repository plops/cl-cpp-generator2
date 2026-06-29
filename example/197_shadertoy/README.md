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
