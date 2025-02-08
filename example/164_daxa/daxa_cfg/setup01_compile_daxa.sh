#!/usr/bin/bash

cmake .. \
      -GNinja \
      -DCMAKE_BUILD_TYPE=Release \
      -DDAXA_INSTALL=ON -DCMAKE_INSTALL_PREFIX=/home/martin/vulkan \
      -DCMAKE_PREFIX_PATH="/home/martin/vulkan;/home/martin/src/imgui" \
      -DDAXA_USE_VCPKG=false \
      -DDAXA_ENABLE_UTILS_FSR2=false \
      -DDAXA_ENABLE_UTILS_IMGUI=false \
      -DDAXA_ENABLE_UTILS_MEM=false \
      -DDAXA_ENABLE_UTILS_PIPELINE_MANAGER_GLSLANG=true \
      -DDAXA_ENABLE_UTILS_PIPELINE_MANAGER_SLANG=false \
      -DDAXA_ENABLE_UTILS_PIPELINE_MANAGER_SPIRV_VALIDATION=false \
      -DDAXA_ENABLE_UTILS_TASK_GRAPH=true \
      -DDAXA_ENABLE_TESTS=true \
      -DDAXA_ENABLE_TOOLS=false \
      -DDAXA_ENABLE_STATIC_ANALYSIS=false \
      -DCMAKE_CXX_STANDARD=20 \
      -DBUILD_SHARED_LIBS=true

#      -DCMAKE_INCLUDE_DIR=/home/martin/src/imgui
