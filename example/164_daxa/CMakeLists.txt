cmake_minimum_required(VERSION 3.6)
project(triangle)
set(CMAKE_CXX_STANDARD 23)

include_directories(~/vulkan/include)

add_executable(
        ${PROJECT_NAME}
        src/main.cpp
)

target_link_libraries(
        ${PROJECT_NAME}
        /home/martin/vulkan/lib64/libdaxa.a
        glfw
        vulkan
        X11
)