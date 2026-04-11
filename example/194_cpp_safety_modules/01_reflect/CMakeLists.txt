cmake_minimum_required(VERSION 3.28)
project(ReflectionModulesExample LANGUAGES CXX)

# GCC 16 requires C++26 for the latest reflection features
set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable experimental reflection in GCC
add_compile_options(-freflection)

add_executable(reflection_demo main.cpp)

# Define the module as a FILE_SET of type CXX_MODULES
target_sources(reflection_demo
    PRIVATE
    FILE_SET CXX_MODULES
    FILES
        ReflectModule.cppm
)
