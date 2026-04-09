#!/usr/bin/env bash

set -euo pipefail

BUILD_PROFILE="${BUILD_PROFILE:-debug}"

CXX="${CXX:-g++}"
SRC="main.cpp"
OUT="forthjit"
COMMON_FLAGS=(
  -std=c++20
  -Wall
  -Wextra
  -pedantic
  -rdynamic
)
LDFLAGS=(
  -lgccjit
)

case "${BUILD_PROFILE}" in
  performance)
    CXXFLAGS=(
      -O3
      -march=native
      -flto
    )
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
echo "Built ./${OUT}"
