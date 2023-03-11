cmake_minimum_required(VERSION 3.1)

project(opencascade_example CXX)

# Set the path to the OpenCASCADE installation directory
set(OCC_DIR "/usr")

# Find the OpenCASCADE libraries
find_package(OpenCASCADE REQUIRED COMPONENTS FoundationClasses ModelingData ) #Visualization)

# Add the OpenCASCADE include directories to the include path
include_directories(${OpenCASCADE_INCLUDE_DIRS} /usr/include/opencascade)

# Add the executable target
add_executable(opencascade_example main.cpp)

# Link the OpenCASCADE libraries to the target
target_link_libraries(opencascade_example ${OpenCASCADE_LIBRARIES})

# Set the compiler options
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
