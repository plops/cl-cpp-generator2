cmake_minimum_required(VERSION 3.0)
project(example LANGUAGES CXX C)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
include_directories(/home/martin/src/sokol /home/martin/src/imgui)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}  -O0  -Wfatal-errors ")
set(SRCS
    /home/martin/stage/cl-cpp-generator2/example/72_emsdk/05source/index.cpp
    /home/martin/stage/cl-cpp-generator2/example/72_emsdk/05source/flextGL.c)
add_executable(index ${SRCS})
if(EMSCRIPTEN)
  set(CMAKE_EXECUTABLE_SUFFIX ".html")
  set_target_properties(
    index PROPERTIES LINK_FLAGS
                     "-s WASM=1 -s LEGACY_GL_EMULATION=1 -s USE_GLFW=3")
else()
  set(CMAKE_C_COMPILER clang)
  set(CMAKE_CXX_COMPILER clang++)
  target_link_libraries(index PRIVATE glfw GL)
endif()
