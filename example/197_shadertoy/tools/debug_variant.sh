#!/bin/sh
# Build a false-colour debug copy of the generated earth shader in /tmp/earth_debug.
#
#   usage: sh tools/debug_variant.sh "<shadeSurface return expr>" ["<final rgb expr>"]
#
# The first expression replaces "return lit;" inside shadeSurface (so any local of
# that function can be probed: isLand, shelf, snow, ice, day, cshadow, ...).
# The optional second expression replaces the final colour of mainImage (locals
# there: col, clouds, tHit, bb, sunDir, ...).  Pass "" to keep a stage unchanged.
set -e
SRC=vulkan-shadertoy-x11/launcher/shaders/earth
DST=/tmp/earth_debug
rm -rf "$DST"
mkdir -p "$DST"
cp "$SRC/common.glsl" "$SRC/buf0.glsl" "$DST/"
python3 - "$SRC/main_image.glsl" "$DST/main_image.glsl" "$1" "${2:-}" <<'PY'
import sys
src, dst, surf, final = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
text = open(src).read()
if surf:
    key = "  return lit;\n"
    assert text.count(key) == 1
    text = text.replace(key, "  return %s;\n" % surf)
if final:
    key = "  fragColor = vec4(col, 1.0F);\n"
    assert text.count(key) == 1, text.count(key)
    text = text.replace(key, "  fragColor = vec4(%s, 1.0);\n" % final)
open(dst, "w").write(text)
PY
echo "debug shader in $DST (surface='$1' final='${2:-}')"
