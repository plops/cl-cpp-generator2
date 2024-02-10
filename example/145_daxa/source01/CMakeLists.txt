cmake_minimum_required(VERSION 3.20)
project(daxa_test CXX)
message(STATUS "${CMAKE_PREFIX_PATH}")

find_package(Vulkan REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(daxa CONFIG REQUIRED)
find_package(VulkanMemoryAllocator CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)
find_package(glslang CONFIG REQUIRED)



file(GLOB sources src/*.cpp)


add_executable(${PROJECT_NAME}
        ${sources}
)

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 20)

target_link_libraries(${PROJECT_NAME}
	daxa::daxa
	glfw
)

