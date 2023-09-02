#!/bin/bash

# Base build directory name
base_dir="build_amd64_debug"

# Sanitizers to be enabled
declare -a sanitizers=("asan" "ubsan" "tsan" "lsan")

# Loop over each sanitizer
for sanitizer in "${sanitizers[@]}"; do
  # Create and enter build directory
  build_dir="${base_dir}_${sanitizer}"
  mkdir -p $build_dir
  cd $build_dir

  # Get dependencies that I already fetched
  cp -ar ../b/_deps .
    
  # Configure with CMake
  cmake_args="-G Ninja -DCMAKE_BUILD_TYPE=Debug -DENABLE_${sanitizer^^}=ON"
  cmake .. $cmake_args
  ninja 
  # Go back to the original directory
  cd ..
done
