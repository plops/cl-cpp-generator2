# CMakeLists.txt
cmake_minimum_required(VERSION 3.10)

project(CharucoBoardCreator)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(OpenCV_DIR "/home/martin/opencv/lib/cmake/opencv4")

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(CharucoBoardCreator main.cpp)

target_link_libraries(CharucoBoardCreator ${OpenCV_LIBS})
