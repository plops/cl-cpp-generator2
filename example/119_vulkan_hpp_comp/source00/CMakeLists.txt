cmake_minimum_required(VERSION 3.10)
project(HelloCompute)

set(CMAKE_CXX_STANDARD 17)

find_package(Vulkan REQUIRED)

add_executable(HelloCompute main.cpp)

target_include_directories(HelloCompute PRIVATE ${Vulkan_INCLUDE_DIRS})

#target_link_libraries(HelloCompute PRIVATE ${Vulkan_LIBRARIES} glfw)



if(MSVC)
    target_compile_options(HelloCompute PRIVATE /W4 /WX)
    target_compile_definitions(HelloCompute PRIVATE VK_USE_PLATFORM_WIN32_KHR)
else()
    target_compile_options(HelloCompute PRIVATE -Wall -Wextra -pedantic -Werror)
endif()

