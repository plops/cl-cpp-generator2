# Minimum required version of CMake
cmake_minimum_required(VERSION 3.14)

# Project name
project(embrace)

# Specify C++ standard (e.g., C++17)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Enable AddressSanitizer
option(ENABLE_ASAN "enable address sanitizer" ON)
if(ENABLE_ASAN)
  add_compile_options(-fsanitize=address)
  add_link_options(-fsanitize=address)
endif()
  
# Add the source file to the project
add_executable(embrace src/main.cpp)
