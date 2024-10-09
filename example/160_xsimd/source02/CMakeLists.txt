cmake_minimum_required(VERSION 3.5)
if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    project(xsimd-examples)

    find_package(xsimd REQUIRED CONFIG)
    set(XSIMD_INCLUDE_DIR ${xsimd_INCLUDE_DIR})
endif ()
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)
include_directories(${XSIMD_INCLUDE_DIR})
#find_package(popl)

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
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -mtune=native")
    endif ()
    if (NOT CMAKE_CXX_COMPILER_ID MATCHES Clang) # We are using clang-cl
 #       set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
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


add_executable(meanSimd src/main.cpp ${XSIMD_HEADERS})
#set_property(TARGET meanSimd PROPERTY CXX_STANDARD 14)
if (ENABLE_XTL_COMPLEX)
    target_link_libraries(meanSimd PRIVATE xtl)
endif ()

add_custom_target(xmeanSimd COMMAND meanSimd DEPENDS meanSimd)
