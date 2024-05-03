cmake_minimum_required(VERSION 3.10)
project(yard)


# GCC 13, CLANG 14 and MSVC 16.10/VS 2019 all have the {fmt} based std::format 

set(CMAKE_CXX_STANDARD 20)

# Enable profiling
option(ENABLE_PROFILING "Enable profiling of C++ code" OFF)

# create an option for compile time profiling (using clang's -ftime-trace)
# open  CMakeFiles/yard.dir/src/main.cpp.json  in chrome://tracing (using google chrome)
# https://aras-p.info/blog/2019/01/16/time-trace-timeline-flame-chart-profiler-for-Clang/
option(ENABLE_COMPILE_TIME_PROFILING "Enable compile time profiling of C++ code" OFF)


# create an option for very fast compilation (turns off optimization)
option(ENABLE_FAST_COMPILE "Enable fast compilation of C++ code" OFF)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Find X11 libraries
find_package(X11 REQUIRED)

# Find all the cpp files in src/
file(GLOB SOURCES "src/*.cpp")

# one can identify the heaviest files using
# https://github.com/nico/ninjatracing
# https://www.reddit.com/r/cpp/comments/hj66pd/c_is_too_slow_to_compile_can_you_share_all_your/

# Add source files
add_executable(${PROJECT_NAME} ${SOURCES})

if(ENABLE_PROFILING)
    # Add compile flags for profiling
    target_compile_options(${PROJECT_NAME} PRIVATE -pg -g3 -Og)
    # Add link flags for profiling
    target_link_options(${PROJECT_NAME} PRIVATE -pg)
endif()

if(ENABLE_COMPILE_TIME_PROFILING)
    # Add compile flags for compile time profiling
    target_compile_options(${PROJECT_NAME} PRIVATE -ftime-trace)
endif()

if(ENABLE_FAST_COMPILE)
    # Add compile flags for fast compilation
    target_compile_options(${PROJECT_NAME} PRIVATE -O0)
endif()

# Link libraries
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${X11_LIBRARIES})

# Add compile flags for optimization
#target_compile_options(${PROJECT_NAME} PRIVATE -Ofast -mfpmath=both -march=native -m64 -funroll-loops -mavx2)

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS} ${X11_INCLUDE_DIR})
