#!/usr/bin/env bash

set -euo pipefail

# Change this to select the build configuration.
BUILD_PROFILE="${BUILD_PROFILE:-performance}"

CXX="g++"
SRC="main.cpp"
OUT="jit_example"
COMMON_FLAGS=(
  -Wall
  -Wextra
)
LDFLAGS=(
  -lgccjit
)

supports_graphite() {
  printf 'int main(){return 0;}\n' | "${CXX}" \
    -x c++ \
    - \
    -S \
    -o /dev/null \
    -fgraphite-identity \
    -floop-nest-optimize \
    >/dev/null 2>&1
}

case "${BUILD_PROFILE}" in
  performance)
    CXXFLAGS=(
      -O3
      -march=native
      -flto
    )
    if supports_graphite; then
      CXXFLAGS+=(
        -fgraphite-identity
        -floop-nest-optimize
      )
    else
      echo "Graphite optimizations unavailable; continuing without them" >&2
    fi
    ;;
  debug)
    CXXFLAGS=(
      -O0
      -g3
      -ggdb
      -fno-omit-frame-pointer
    )
    ;;
  size)
    CXXFLAGS=(
      -Os
      -s
      -march=native
    )
    ;;
  *)
    echo "Unknown BUILD_PROFILE: ${BUILD_PROFILE}" >&2
    echo "Use one of: performance, debug, size" >&2
    exit 1
    ;;
esac

echo "Compiling with BUILD_PROFILE=${BUILD_PROFILE}"
"${CXX}" "${COMMON_FLAGS[@]}" "${CXXFLAGS[@]}" "${SRC}" -o "${OUT}" "${LDFLAGS[@]}"
"./${OUT}"
