# CMake version
cmake_minimum_required(VERSION 3.10)

# Project name
project(imgui_sdl2_bullet_gears_designer)

# C++ Standard
set(CMAKE_CXX_STANDARD 20)

# Add SDL2
find_package(SDL2 REQUIRED)
include_directories(${SDL2_INCLUDE_DIRS})

# Add OpenGL
find_package(OpenGL REQUIRED)
include_directories(${OPENGL_INCLUDE_DIR})

# Add ImGui
include_directories(
  ${CMAKE_SOURCE_DIR}/imgui
  ${CMAKE_SOURCE_DIR}/imgui/backends
)

# Define the executable
add_executable(${PROJECT_NAME}
    ${CMAKE_SOURCE_DIR}/src/main.cpp 
   	${CMAKE_SOURCE_DIR}/imgui/imgui.cpp
    ${CMAKE_SOURCE_DIR}/imgui/imgui_draw.cpp
    ${CMAKE_SOURCE_DIR}/imgui/backends/imgui_impl_sdl2.cpp
    ${CMAKE_SOURCE_DIR}/imgui/backends/imgui_impl_opengl3.cpp
    ${CMAKE_SOURCE_DIR}/imgui/imgui_widgets.cpp
)

# Linking libraries
target_link_libraries(${PROJECT_NAME}
	${SDL2_LIBRARIES}
	${OPENGL_LIBRARIES})
