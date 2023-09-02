#!/bin/bash


# Release build without sanitizers
mkdir -p build_amd64_release
cd build_amd64_release
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DFETCHCONTENT_BASE_DIR=../fetchcontent
ninja
cd ..


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
    
  # Configure with CMake
  cmake_args="-G Ninja -DCMAKE_BUILD_TYPE=Debug -DFETCHCONTENT_BASE_DIR=../fetchcontent  -DENABLE_${sanitizer^^}=ON"
  cmake .. $cmake_args
  ninja 
  # Go back to the original directory
  cd ..
done
