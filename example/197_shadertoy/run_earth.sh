#!/bin/sh
# =========================================================================
# RUN_EARTH.SH -- animated Earth globe shader, headless GPU rendering
# =========================================================================
#   1. (optional) re-bake the Earth datasets   -> earth_data.lisp
#   2. transpile the Lisp shader description   -> shaders/earth/*.glsl
#   3. build the headless EGL runner
#   4. render screenshots of the marker tour + a benchmark
#
# Needs: sbcl + quicklisp (cl-cpp-generator2), g++, libEGL/libGL/libpng,
#        an NVIDIA GPU reachable through EGL (see doc/headless.md).
# =========================================================================
set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR"

SHADERS=vulkan-shadertoy-x11/launcher/shaders/earth

if [ "$1" = "--bake" ]; then
    echo "=== Step 0: baking Earth datasets (needs network on first run) ==="
    python3 tools/bake_earth_data.py
fi

echo "=== Step 1: transpiling gen4.lisp -> GLSL ==="
sbcl --noinform --non-interactive --load gen4.lisp

echo "=== Step 2: building the headless EGL runner ==="
g++ -O3 headless_gpu_runner.cpp -lEGL -lGL -lpng -o headless_gpu_runner

echo "=== Step 3: rendering the tour (one screenshot per marker) ==="
# The state pass is a pure function of iTime, so --time jumps straight to a
# moment of the tour instead of stepping through hundreds of frames.
i=0
for t in 2 6 10 14 18 22; do
    out=$(printf "earth_tour_%02d_t%02ds.png" "$i" "$t")
    ./headless_gpu_runner --res 1280 720 --frames 2 --time "$t" \
        --shader-dir "$SHADERS" --screenshot "$out" | tail -1
    i=$((i + 1))
done

echo "=== Step 4: 1080p benchmark ==="
./headless_gpu_runner --res 1920 1080 --frames 2 --time 2 --benchmark 400 \
    --shader-dir "$SHADERS" --screenshot earth_marker_berlin_1080p.png \
    | sed -n '/BENCHMARK/,$p'

echo
echo "Optional: animated GIF of two tour segments"
echo "  python3 tools/make_animation.py earth_tour.gif 0 8 48 480 300"
