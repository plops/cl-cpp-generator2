cmake_minimum_required(VERSION 3.10)
project(yard)


# GCC 13, CLANG 14 and MSVC 16.10/VS 2019 all have the {fmt} based std::format 

set(CMAKE_CXX_STANDARD 20)

# Enable profiling
option(ENABLE_PROFILING "Enable profiling of C++ code" OFF)


# Find OpenCV
find_package(OpenCV REQUIRED)

# Find X11 libraries
find_package(X11 REQUIRED)

# Find all the cpp files in src/
file(GLOB SOURCES "src/*.cpp")

# Add source files
add_executable(${PROJECT_NAME} ${SOURCES})

if(ENABLE_PROFILING)
    # Add compile flags for profiling
    target_compile_options(${PROJECT_NAME} PRIVATE -pg -g3 -Og)
    # Add link flags for profiling
    target_link_options(${PROJECT_NAME} PRIVATE -pg)
endif()


# Link libraries
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${X11_LIBRARIES})

# Add compile flags for optimization
#target_compile_options(${PROJECT_NAME} PRIVATE -Ofast -mfpmath=both -march=native -m64 -funroll-loops -mavx2)

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS} ${X11_INCLUDE_DIR})
