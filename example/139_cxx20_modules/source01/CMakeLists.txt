# run with:
# mkdir b; cd b; export CXX=/usr/lib/llvm/16/bin/clang++-16; cmake -GNinja .. 

cmake_minimum_required(VERSION 3.26)
project(std_module_example CXX)

# CMake 3.25
#set(CMAKE_EXPERIMENTAL_CXX_MODULE_CMAKE_API "3c375311-a3c9-4396-a187-3227ef642046")
# CMake 3.26 
set(CMAKE_EXPERIMENTAL_CXX_MODULE_CMAKE_API "2182bf5c-ef0d-489a-91da-49dbc3090d2a")

# Default to C++ extensions being off. Clang's modules support have trouble
# with extensions right now and it is not required for any other compiler
set(CMAKE_CXX_EXTENSIONS OFF)

#if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
#  include(gcc_modules.cmake)
#endif()
set(CMAKE_CXX_STANDARD 20)
add_library(q)
target_sources(q
  PUBLIC
    FILE_SET cxx_modules TYPE CXX_MODULES FILES
    src/Vector.cpp
)
add_executable(hello src/user.cpp)
target_link_libraries(hello PRIVATE q)
