#!/usr/bin/env python3
"""Assemble the same fragment shader source the headless runner builds.

Mirrors build_shadertoy_frag() in headless_gpu_runner.cpp so that GLSL compiler
line numbers can be mapped back to real source lines.

usage: assemble_frag.py <shader-dir> <buf0|main_image> [out.frag]
"""
import os
import sys

HEADER = """#version 330 core
#extension GL_ARB_explicit_attrib_location : enable
out vec4 FragColor;
in vec2 v_uv;
uniform vec3 iResolution;
uniform float iTime;
uniform float iTimeDelta;
uniform int iFrame;
uniform vec4 iMouse;
uniform vec4 iDate;
uniform sampler2D iChannel0;
uniform sampler2D iChannel1;
uniform sampler2D iChannel2;
uniform sampler2D iChannel3;
uniform sampler2D iKeyboard;

// --- COMMON ---
"""

FOOTER = """
void main() {
    vec4 col = vec4(0.0);
    vec2 fragCoord = gl_FragCoord.xy;
    mainImage(col, fragCoord);
    FragColor = col;
}
"""


def read(path):
    return open(path).read() if os.path.exists(path) else ""


def main():
    sdir, which = sys.argv[1], sys.argv[2]
    out = sys.argv[3] if len(sys.argv) > 3 else f"/tmp/full_{which}.frag"
    common = read(os.path.join(sdir, "common.glsl"))
    body = read(os.path.join(sdir, f"{which}.glsl"))
    src = HEADER + common + "\n\n// --- SHADER BODY ---\n" + body + "\n" + FOOTER
    open(out, "w").write(src)
    print(f"wrote {out} ({len(src.splitlines())} lines)")


if __name__ == "__main__":
    main()
