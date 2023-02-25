cmake_minimum_required(VERSION 3.0)

project(my_project)

# Find Boost library
find_package(fmt REQUIRED)

# Add executable
add_executable(my_exe main.cpp)

# Link with Boost libraries
target_link_libraries(my_exe fmt::fmt)
