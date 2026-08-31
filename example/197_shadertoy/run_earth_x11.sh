#!/bin/sh
# =========================================================================
# RUN_EARTH_X11.SH -- animated Earth globe shader in an X11 Vulkan window
# =========================================================================
# Regenerates the Earth GLSL, builds its Vulkan SPIR-V wrappers, stages the
# display-capable XCB/Vulkan renderer, and opens the interactive window.
#
# Needs: an authorized X11 $DISPLAY, sbcl + quicklisp (cl-cpp-generator2),
# glslangValidator, gcc, Vulkan development libraries, and an XCB/Vulkan GPU.
# Pass renderer options through, for example:
#   ./run_earth_x11.sh --resX 1920 --resY 1080
# =========================================================================
set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR"

if [ -z "${DISPLAY:-}" ]; then
    echo "Error: DISPLAY is unset; run this script from an authorized X11 session." >&2
    exit 1
fi

if [ "$#" -eq 0 ]; then
    set -- --resX 1280 --resY 720
fi

echo "=== Step 1: transpiling gen4.lisp -> Earth GLSL ==="
sbcl --noinform --non-interactive --load gen4.lisp

echo "=== Step 2: compiling Vulkan shader wrappers ==="
(
    cd vulkan-shadertoy-x11/launcher/shaders
    sh build_shaders.sh
)

echo "=== Step 3: building and staging the X11 Vulkan renderer ==="
(
    cd vulkan-shadertoy-x11/build_scripts/build_linux_x11
    sh build.sh
)

echo "=== Step 4: starting the Earth shader window ==="
cd vulkan-shadertoy-x11/build_scripts/build_linux_x11
exec ./VK_shadertoy "$@"
