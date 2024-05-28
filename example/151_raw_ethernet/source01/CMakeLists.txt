# Set the minimum required CMake version
cmake_minimum_required(VERSION 3.10)

# Set the project name
project(packet_listener)

# Specify C++17 standard
set(CMAKE_CXX_STANDARD 23)

# Find necessary libraries (this example is minimal, you might need more)
# You can use find_package for more complex dependencies.
#find_package(Threads REQUIRED)

# Add the executable
add_executable(packet_listener src/main.cpp)

# Link libraries
#target_link_libraries(packet_listener Threads::Threads) 

# Add compiler options (optional)
# For example, to enable warnings:
target_compile_options(packet_listener PRIVATE -Wall -Wextra)
