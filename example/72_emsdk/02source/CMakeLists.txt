cmake_minimum_required(VERSION 3.0)
project(example LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(OpenCV_DIR /home/martin/src/opencv/build_wasm/)
set(OpenCV_STATIC ON)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS}
                    /home/martin/src/opencv_contrib/modules/aruco/include)
option(BUILD_WASM "Build Webassembly" ON)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_FLAGS_DEBUG
    "${CMAKE_CXX_FLAGS_DEBUG}  -ggdb -O0    -Wall -Wextra -Wno-cast-align -Wcast-qual -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2 -Winit-self  -Wmissing-declarations -Wmissing-include-dirs  -Woverloaded-virtual -Wredundant-decls -Wshadow  -Wswitch-default -Wundef -Werror  -Wno-unused -Wno-unused-parameter "
)
set(CMAKE_EXECUTABLE_SUFFIX ".html")
set(SRCS
    /home/martin/stage/cl-cpp-generator2/example/72_emsdk/02source/Charuco.cpp
    /home/martin/stage/cl-cpp-generator2/example/72_emsdk/02source/index.cpp
    /home/martin/src/opencv_contrib/modules/aruco/src/apriltag_quad_thresh.cpp
    /home/martin/src/opencv_contrib/modules/aruco/src/aruco.cpp
    /home/martin/src/opencv_contrib/modules/aruco/src/charuco.cpp
    /home/martin/src/opencv_contrib/modules/aruco/src/dictionary.cpp
    /home/martin/src/opencv_contrib/modules/aruco/src/zmaxheap.cpp)
add_executable(index ${SRCS})
target_link_libraries(index ${OpenCV_LIBS})
