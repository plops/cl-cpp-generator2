# Minimum required version of CMake
cmake_minimum_required(VERSION 3.14)

# Project name
project(run_01_base)

# Specify C++ standard (e.g., C++17)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Enable AddressSanitizer
option(ENABLE_ASAN "enable address sanitizer" ON)
if(ENABLE_ASAN)
  add_compile_options(-fsanitize=address)
  add_link_options(-fsanitize=address)
endif()
  

# Find the GLFW3 and Vulkan packages
find_package(PkgConfig REQUIRED)
pkg_check_modules(GLFW3 REQUIRED glfw3)
find_library(VULKAN_LIBRARY vulkan)

# Add the source file to the project
add_executable(run_01_base run_01_base.cpp)

# Link GLFW3, Vulkan libraries
target_link_libraries(run_01_base ${GLFW3_LIBRARIES} ${VULKAN_LIBRARY})

