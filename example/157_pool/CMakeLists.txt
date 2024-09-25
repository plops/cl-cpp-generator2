# Minimum required version of CMake
cmake_minimum_required(VERSION 3.14)

# Project name
project(Point2DMemoryResourceExample)

# Specify C++ standard (e.g., C++17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add the source file to the project
add_executable(Point2DMemoryResourceExample pmr.cpp)
