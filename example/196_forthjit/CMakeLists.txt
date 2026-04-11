cmake_minimum_required(VERSION 3.10)
project(ForthJIT)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find GCC version to get libgccjit paths
execute_process(COMMAND gcc -dumpversion OUTPUT_VARIABLE GCC_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND gcc -dumpmachine OUTPUT_VARIABLE GCC_MACHINE OUTPUT_STRIP_TRAILING_WHITESPACE)

set(GCCJIT_INCLUDE_DIR "/usr/lib/gcc/${GCC_MACHINE}/${GCC_VERSION}/include")
set(GCCJIT_LIB_DIR "/usr/lib/gcc/${GCC_MACHINE}/${GCC_VERSION}")

include_directories(${GCCJIT_INCLUDE_DIR})
link_directories(${GCCJIT_LIB_DIR})

add_executable(forthjit
    main.cpp
    Operation.cpp
    JITCompiler.cpp
    ForthVM.cpp
    ForthVM_wrappers.cpp
)

target_link_libraries(forthjit gccjit)
