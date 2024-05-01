
# Find X11 libraries
find_package(X11 REQUIRED)

# Add source files
add_executable(screen_capture main.cpp)

# Link libraries
target_link_libraries(screen_capture ${OpenCV_LIBS} ${X11_LIBRARIES})

# Add compile flags for optimization
target_compile_options(screen_capture PRIVATE -Ofast -mfpmath=both -march=native -m64 -funroll-loops -mavx2)

# Include directories
target_include_directories(screen_capture PRIVATE ${OpenCV_INCLUDE_DIRS} ${X11_INCLUDE_DIR})
