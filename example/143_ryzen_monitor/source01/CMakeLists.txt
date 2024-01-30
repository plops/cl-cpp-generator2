
# Example usage:
#  mkdir build
#  cd build
#  cmake -G Ninja -DCMAKE_BUILD_TYPE=Release ..

cmake_minimum_required(VERSION 3.28)
project(ryzen_mon_glgui CXX C)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "" FORCE)
endif()

set(CMAKE_CXX_STANDARD 20)

# GLFW
set(GLFW_DIR /home/martin/src/glfw) # Set this to point to an up-to-date GLFW repo
option(GLFW_BUILD_EXAMPLES "Build the GLFW example programs" OFF)
option(GLFW_BUILD_TESTS "Build the GLFW test programs" OFF)
option(GLFW_BUILD_DOCS "Build the GLFW documentation" OFF)
option(GLFW_INSTALL "Generate installation target" OFF)
option(GLFW_DOCUMENT_INTERNALS "Include internals in documentation" OFF)
add_subdirectory(${GLFW_DIR} binary_dir EXCLUDE_FROM_ALL)
include_directories(${GLFW_DIR}/include)

# Dear ImGui
set(IMGUI_DIR /home/martin/src/imgui)
include_directories(${IMGUI_DIR} ${IMGUI_DIR}/backends ..)

# ImPlot
set(IMPLOT_DIR /home/martin/src/implot)
include_directories(${IMPLOT_DIR})

# Ryzen
set(RYZEN_DIR /home/martin/src/ryzen_monitor/src)
include_directories(${RYZEN_DIR} ${RYZEN_DIR}/lib)

# Libraries
find_package(OpenGL REQUIRED)
set(LIBRARIES "glfw;OpenGL")

include_directories(${GLFW_DIR}/deps)

set(sources src/main.cpp src/CpuAffinityManager.cpp)

#file(GLOB sources src/*.cpp)

add_executable(${PROJECT_NAME}
	${sources}
	${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
	${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
	${IMGUI_DIR}/imgui.cpp
	${IMGUI_DIR}/imgui_draw.cpp
	${IMGUI_DIR}/imgui_demo.cpp
	${IMGUI_DIR}/imgui_tables.cpp
	${IMGUI_DIR}/imgui_widgets.cpp
	${IMPLOT_DIR}/implot.cpp
	${IMPLOT_DIR}/implot_items.cpp
	${IMPLOT_DIR}/implot_demo.cpp
	${RYZEN_DIR}/lib/libsmu.c
	${RYZEN_DIR}/pm_tables.c
	${RYZEN_DIR}/readinfo.c
	${RYZEN_DIR}/ryzen_monitor.c
	)
target_link_libraries(${PROJECT_NAME} ${LIBRARIES})
target_compile_definitions(${PROJECT_NAME} PUBLIC -DImTextureID=ImU64)
