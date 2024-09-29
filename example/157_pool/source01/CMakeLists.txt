# Minimum required version of CMake
cmake_minimum_required(VERSION 3.14)

# Project name
project(pmrTest)

# Specify C++ standard (e.g., C++17)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Enable AddressSanitizer
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address")
endif()
  
# Add the source file to the project
add_executable(pmrTest src/main.cpp)
