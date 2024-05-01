cmake_minimum_required(VERSION 3.10)
project(screen_capture)


# GCC 13, CLANG 14 and MSVC 16.10/VS 2019 all have the {fmt} based std::format 

set(CMAKE_CXX_STANDARD 20)


# Find OpenCV
find_package(OpenCV REQUIRED)

# Find X11 libraries
find_package(X11 REQUIRED)

# Add source files
add_executable(screen_capture src/Screenshot.h src/Screenshot.cpp src/main.cpp)

# Link libraries
target_link_libraries(screen_capture ${OpenCV_LIBS} ${X11_LIBRARIES})

# Add compile flags for optimization
target_compile_options(screen_capture PRIVATE -Ofast -mfpmath=both -march=native -m64 -funroll-loops -mavx2)

# Include directories
target_include_directories(screen_capture PRIVATE ${OpenCV_INCLUDE_DIRS} ${X11_INCLUDE_DIR})
