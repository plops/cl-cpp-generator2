cmake_minimum_required(VERSION 3.16)
project(YourProjectName)

set(CMAKE_CXX_STANDARD 17)

# location of IGL library
set(IGL_DIR "/home/martin/igl/")

# I assume the path "/home/martin/igl/include" has the necessary include files for IGL
include_directories(${IGL_DIR}/include)

add_executable(YourProjectName main.cpp)

# I assume the necessary IGL, tinyobjloader and meshoptimizer libraries are in "/home/martin/igl/lib"
find_library(IGLLIBRARY IGLLibrary HINTS ${IGL_DIR}/lib)
find_library(TINYOBJLOADER tinyobjloader HINTS ${IGL_DIR}/lib)
find_library(MESHOPTIMIZER meshoptimizer HINTS ${IGL_DIR}/lib)

target_link_libraries(YourProjectName PRIVATE ${IGLLIBRARY} ${TINYOBJLOADER} ${MESHOPTIMIZER})

# Additional library dependencies as per your setup
if(UNIX)
  target_link_libraries(YourProjectName PRIVATE EGL)
endif()

# include GLFW if it's installed in the system
find_package(glfw3 REQUIRED)
target_link_libraries(YourProjectName PRIVATE glfw)
