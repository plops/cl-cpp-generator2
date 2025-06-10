cmake_minimum_required(VERSION 3.16) # FetchContent benefits from newer versions
project(NetworkParserKoan CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF) # Prefer not to use GNU extensions if not needed

# Optional: Compiler flags
if (CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_CLANG)
    target_compile_options(variant PRIVATE -Wall -Wextra -Wpedantic) # -Werror
    target_compile_options(unit_tests PRIVATE -Wall -Wextra -Wpedantic)
endif ()

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(
            -g -g3 -ggdb -gdwarf-4
            -Og
    )
    add_link_options(-Wl,--print-memory-usage,--print-gc-sections,-Map=${PROJECT_BINARY_DIR}/${PROJECT_NAME}.map,--cref)
else ()

    #set(CMAKE_UNITY_BUILD true) # merge c++ files
    add_compile_options(
            #-O0
            #-Os
            -O3 -march=native
            -ffunction-sections -fdata-sections # don't use together with flto
            #-g -g3 -ggdb -gdwarf-4   # useful if you want line numbers in *.nm output
    )
    #Cset(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE) # flto


    #   -gc-sections: Enables "garbage collection" at the linker stage. The linker will analyze your code and discard any unused functions or data sections to reduce the final executable size.
    #   --print-gc-sections: Instructs the linker to generate a report detailing which sections were removed during the garbage collection process. This can help you pinpoint potential areas for further size optimization.
    #   --print-memory-usage: Tells the linker to provide a summary of the memory usage of your executable, broken down into categories like code, data, etc. This helps analyze memory consumption.
    #   -Map=...: Generates a map file that details the memory layout of your linked executable.
    #   --cref: Enables the generation of a cross-reference table. This table provides information about how symbols (functions, variables) are defined and used throughout your codebase.

    add_link_options(-Wl,-gc-sections,--print-memory-usage,--print-gc-sections,-Map=${PROJECT_BINARY_DIR}/${PROJECT_NAME}.map,--cref)
endif ()


# Koan: Locating the Scrolls - Defining Source Files
set(PROTOCOL_H src/protocol.h)
set(PARSER_H src/parser.h)
set(PARSER_CPP src/parser.cpp)
set(SERVER_CPP src/server.cpp)

set(PARSER_LIB_SOURCES ${PARSER_CPP} ${PARSER_H} ${PROTOCOL_H})

find_package(GTest)
find_package(benchmark)

# Koan: A Library of Wisdom - The Parser Logic as a Library
# This makes it reusable for the server, tests, and benchmarks.
add_library(ParserLib STATIC ${PARSER_LIB_SOURCES})
target_include_directories(ParserLib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)


# Koan: The Server's Manifest - Building the Network Server Executable
add_executable(network_server ${SERVER_CPP})
target_link_libraries(network_server PRIVATE ParserLib)
# On some systems, POSIX threads might be needed for socket functions or future expansion
# find_package(Threads REQUIRED)
# target_link_libraries(network_server PRIVATE Threads::Threads)

# Koan: The Trials - Setting up Google Test
# We use FetchContent to download and build GoogleTest as part of our project.
#include(FetchContent)
#FetchContent_Declare(
#        googletest
#        URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip # Using a known stable tag
#        # Or use GIT_REPOSITORY and GIT_TAG for more flexibility
#        # GIT_REPOSITORY https://github.com/google/googletest.git
#        # GIT_TAG release-1.14.0
#)
## Set QUIET to avoid too much FetchContent noise during configure
#FetchContent_MakeAvailable(googletest)

add_executable(parser_tests tests/parser_test.cpp)
target_link_libraries(parser_tests PRIVATE ParserLib gtest_main gtest) # gtest_main links gtest and provides main()

add_executable(network_client src/client.cpp src/test_utils.cpp)
#target_link_libraries(network_client PRIVATE BufferCore)


# Automatically discover and add tests to CTest
include(GoogleTest)
gtest_discover_tests(parser_tests)


# Koan: The Measurements - Setting up Google Benchmark
#FetchContent_Declare(
#        googlebenchmark
#        URL https://github.com/google/benchmark/archive/refs/tags/v1.8.0.zip # Using a known stable tag
#        # GIT_REPOSITORY https://github.com/google/benchmark.git
#        # GIT_TAG v1.8.0
#)
#FetchContent_MakeAvailable(googlebenchmark)

add_executable(parser_benchmarks benchmarks/parser_benchmark.cpp)
# benchmark::benchmark links both benchmark and benchmark_main
target_link_libraries(parser_benchmarks PRIVATE ParserLib benchmark::benchmark)

# Koan: The Path of Execution
# After building with CMake and your chosen build system (e.g., make, ninja):
#
# To run the server:
#   ./network_server
#
# To run the unit tests:
#   ./parser_tests
#
# To run the microbenchmarks:
#   ./parser_benchmarks
#
# Ponder upon the results, and may your understanding of C++ deepen.
