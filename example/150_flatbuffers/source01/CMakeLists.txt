cmake_minimum_required(VERSION 3.10)

project(flatbuffers_image)

# Find the FlatBuffers library
find_package(Flatbuffers REQUIRED)

# Include FlatBuffers generated headers
include_directories(${FLATBUFFERS_INCLUDE_DIRS})

# Add executable
add_executable(image_serialize src/main.cpp src/image_generated.h)

# Link against the FlatBuffers library
target_link_libraries(image_serialize ${FLATBUFFERS_LIBRARIES})
