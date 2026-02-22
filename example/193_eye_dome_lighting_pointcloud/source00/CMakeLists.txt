cmake_minimum_required(VERSION 3.10)
project(PointcloudEDL CXX)

# Enforce Modern C++ Standards
set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Expand the home directory path to locate dependencies
set(SRC_DIR "$ENV{HOME}/src")

# 1. GLM (Header-only mathematics library)
add_library(glm INTERFACE)
target_include_directories(glm INTERFACE ${SRC_DIR}/glm)

# 2. GLAD (OpenGL loader generated via web interface)
add_library(glad STATIC ${SRC_DIR}/glad/src/gl.c)
target_include_directories(glad PUBLIC ${SRC_DIR}/glad/include)

# 3. GLFW (Window and context management)
# We disable GLFW's tests and documentation to vastly accelerate build times
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
add_subdirectory(${SRC_DIR}/glfw glfw_build)

# 4. Dear ImGui (Immediate mode graphical user interface)
set(IMGUI_DIR ${SRC_DIR}/imgui)
add_library(imgui STATIC
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
)
target_include_directories(imgui PUBLIC 
    ${IMGUI_DIR} 
    ${IMGUI_DIR}/backends
)
# ImGui backends require linking against GLFW
target_link_libraries(imgui glfw)

# Find System OpenGL dynamically based on the Linux distribution
find_package(OpenGL REQUIRED)

# Define Main Executable
add_executable(PointcloudEDL main.cpp)

# Link all dependencies to the main executable
# CMAKE_DL_LIBS is strictly required on Linux for GLAD to dynamically load OpenGL pointers
target_link_libraries(PointcloudEDL PRIVATE 
    glad 
    glfw 
    glm 
    imgui 
    OpenGL::GL 
    ${CMAKE_DL_LIBS} 
)
