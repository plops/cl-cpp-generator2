# generated with qwen2.5
cmake_minimum_required(VERSION 3.14)
project(VulkanGLFWApp)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find packages
find_package(PkgConfig REQUIRED)
find_package(Vulkan REQUIRED)

pkg_check_modules(GLFW REQUIRED glfw3)
pkg_check_modules(GLM REQUIRED glm)

# Include directories for GLFW and GLM
include_directories(${GLFW_INCLUDE_DIRS})
include_directories(${GLM_INCLUDE_DIRS})

# Add the executable target
add_executable(VulkanGLFWApp main.cpp)

# Link libraries
target_link_libraries(VulkanGLFWApp ${GLFW_LIBRARIES} ${GLM_LIBRARIES} Vulkan::Vulkan)

# Optionally, you can add a command to clean up Vulkan resources on application exit
set_target_properties(VulkanGLFWApp PROPERTIES
    CLEAN_DIRECTORIES "${CMAKE_BINARY_DIR}/vulkan"
)
