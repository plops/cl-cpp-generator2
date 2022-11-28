cmake_minimum_required(VERSION 3.0)
project(example LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
include_directories(/home/martin/src/sokol /home/martin/src/imgui)
option(BUILD_WASM "Build Webassembly" ON)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}  -ggdb -O0    ")
set(CMAKE_EXECUTABLE_SUFFIX ".html")
set(SRCS
    /home/martin/stage/cl-cpp-generator2/example/72_emsdk/03source/SokolApp.cpp
    /home/martin/stage/cl-cpp-generator2/example/72_emsdk/03source/SokolAppImpl.cpp
    /home/martin/stage/cl-cpp-generator2/example/72_emsdk/03source/index.cpp
    /home/martin/stage/cl-cpp-generator2/example/72_emsdk/03source/sokolapp.cpp
    /home/martin/stage/cl-cpp-generator2/example/72_emsdk/03source/sokolappImpl.cpp
    /home/martin/src/imgui/imgui.cpp
    /home/martin/src/imgui/imgui_demo.cpp
    /home/martin/src/imgui/imgui_draw.cpp
    /home/martin/src/imgui/imgui_tables.cpp
    /home/martin/src/imgui/imgui_widgets.cpp)
add_executable(index ${SRCS})
