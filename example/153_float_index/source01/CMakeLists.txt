# Set the minimum required CMake version
cmake_minimum_required(VERSION 3.10)

# Set the project name
project(float_index)

# Specify C++17 standard
set(CMAKE_CXX_STANDARD 23)

# Add the executable
add_executable(${PROJECT_NAME} src/main.cpp)

# Add compiler options (optional)
# For example, to enable warnings:
target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra)
