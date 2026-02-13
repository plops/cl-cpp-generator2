cmake_minimum_required(VERSION 3.25)
project(GrpcPluginDemo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler Flags
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_compile_options(-Wall -Wextra -Wpedantic)
    
    if(USE_SANITIZERS)
        message(STATUS "Enabling Sanitizers: address, leak, undefined")
        add_compile_options(-fsanitize=address,leak,undefined -fno-omit-frame-pointer)
        add_link_options(-fsanitize=address,leak,undefined)
    endif()

    if(USE_LTO)
        message(STATUS "Enabling LTO")
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
    endif()

    if(USE_GRAPHITE)
        message(STATUS "Enabling Graphite Optimizations")
        add_compile_options(-fgraphite-identity -floop-nest-optimize)
    endif()
    
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        message(STATUS "Release build: keeping debug symbols")
        add_compile_options(-g)
    endif()
endif()

# Dependencies
find_package(Protobuf REQUIRED)
find_package(gRPC REQUIRED)

# Includes
include_directories(include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}) # for generated proto headers

# Targets will be added here
# add_subdirectory(src/proto)
# add_subdirectory(src/alpha)
# add_subdirectory(src/beta)
# add_subdirectory(src/libone)
# add_subdirectory(src/libtwo)
