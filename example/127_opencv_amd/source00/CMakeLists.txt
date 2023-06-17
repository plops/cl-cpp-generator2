# CMakeLists.txt
cmake_minimum_required(VERSION 3.10)

project(OpenCLCheck)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(OpenCLCheck main.cpp)

target_link_libraries(OpenCLCheck ${OpenCV_LIBS}) # needs vtk and hdf5
# target_link_libraries(OpenCLCheck ${OpenCV_core_LIBS} ${OpenCV_highgui_LIBS})
