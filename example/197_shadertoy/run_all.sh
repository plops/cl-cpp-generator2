#!/bin/sh
# =========================================================================
# RUN_ALL.SH - Build and Run Interactive Vulkan Shadertoy Raymarching
# =========================================================================
# This script automates:
#   1. Running the Common Lisp transpiler to generate GLSL shaders.
#   2. Compiling the generated GLSL shaders to SPIR-V using glslang.
#   3. Compiling the native X11 Vulkan launcher using GCC.
#   4. Executing the compiled launcher.
# =========================================================================

set -e

# Get script's parent directory path
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR"

echo "=== Step 1: Running Lisp Transpiler ==="
sbcl --load gen3.lisp --quit

echo "=== Step 2: Compiling Shaders to SPIR-V ==="
cd vulkan-shadertoy-x11/launcher/shaders/
sh build_shaders.sh

echo "=== Step 3: Compiling VK_shadertoy Launcher ==="
cd ../../build_scripts/build_linux_x11/
sh build.sh

echo "=== Step 4: Running VK_shadertoy ==="
./VK_shadertoy
