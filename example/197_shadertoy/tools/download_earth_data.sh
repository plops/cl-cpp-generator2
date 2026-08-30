#!/bin/sh
# Download the external Earth inputs used by bake_earth_data.py.
# The files stay in tools/cache/ (ignored by git) and are regenerated on demand.
#
# Sources:
#   Natural Earth 110m land polygons (public domain)
#   NASA Blue Marble land_shallow_topo_2048
#   NASA Earth's City Lights
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
CACHE_DIR=${EARTH_DATA_CACHE_DIR:-"$SCRIPT_DIR/cache"}
FORCE=0
if [ "${1:-}" = "--force" ]; then
    FORCE=1
    shift
fi
if [ "$#" -ne 0 ]; then
    echo "usage: $0 [--force]" >&2
    exit 2
fi

mkdir -p "$CACHE_DIR"

fetch() {
    name=$1
    url=$2
    target="$CACHE_DIR/$name"
    if [ "$FORCE" -eq 0 ] && [ -s "$target" ]; then
        echo "cached: $name"
        return
    fi
    echo "download: $name"
    curl --fail --location --retry 3 --silent --show-error "$url" --output "$target"
}

fetch ne_110m_land.geojson \
  https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_land.geojson
fetch blue_marble.jpg \
  https://eoimages.gsfc.nasa.gov/images/imagerecords/57000/57752/land_shallow_topo_2048.jpg
fetch night_lights.jpg \
  https://eoimages.gsfc.nasa.gov/images/imagerecords/55000/55167/earth_lights_lrg.jpg

echo "Earth inputs are available in $CACHE_DIR"
