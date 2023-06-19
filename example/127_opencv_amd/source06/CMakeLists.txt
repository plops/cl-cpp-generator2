cmake_minimum_required(VERSION 3.10)
project(DetectArucoInVideo)

set(CMAKE_CXX_STANDARD 17)

# Where to find CMake modules and OpenCV
set(OpenCV_DIR "/home/martin/opencv/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)

# Declare the executable target built from your sources
add_executable(DetectArucoInVideo main.cpp)

# Link your application with OpenCV libraries
target_link_libraries(DetectArucoInVideo PRIVATE ${OpenCV_LIBS})
