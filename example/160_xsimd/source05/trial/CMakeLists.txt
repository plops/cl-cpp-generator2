cmake_minimum_required(VERSION 3.5)
project(thread_merge)

set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)

#include_directories(/usr/include/eigen3)
#find_package(popl)
#find_package(Eigen3)

if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "Setting examples build type to Release")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
else ()
    message(STATUS "Tests build type is ${CMAKE_BUILD_TYPE}")
endif ()

if (CMAKE_BUILD_TYPE STREQUAL Debug)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb3 -O0")
endif ()

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    if (NOT CMAKE_CXX_FLAGS MATCHES "-march" AND NOT CMAKE_CXX_FLAGS MATCHES "-arch" AND NOT CMAKE_OSX_ARCHITECTURES)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -mtune=native -ggdb3")
    endif ()
    if (NOT CMAKE_CXX_COMPILER_ID MATCHES Clang) # We are using clang-cl
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
    endif ()
endif ()

# Specify C++ standard (e.g., C++17)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Enable AddressSanitizer
option(ENABLE_ASAN "enable address sanitizer" OFF)
if (ENABLE_ASAN)
    add_compile_options(-fsanitize=address)
    add_link_options(-fsanitize=address)
endif ()


add_executable(${PROJECT_NAME} thread_merge.cpp)
