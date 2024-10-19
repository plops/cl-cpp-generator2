# i don't have glew and implot

# Example usage:
#  mkdir build
#  cd build
#  cmake -G Ninja -DCMAKE_BUILD_TYPE=Release ..

cmake_minimum_required(VERSION 3.27)
project(ryzen_mon_glgui CXX C)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "" FORCE)
endif ()

if (CMAKE_BUILD_TYPE STREQUAL "Release")
    # enable link time optimization
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    add_compile_options(-march=native)
endif ()

# GCC 13, CLANG 14 and MSVC 16.10/VS 2019 all have the {fmt} based std::format 

set(CMAKE_CXX_STANDARD 20)


# Set DEP_DIR with a command line flag
if (NOT DEP_DIR)
    set(DEP_DIR "/tmp/deps143/" CACHE PATH "Path to dependencies directory")
endif ()

# Create a CMake Option to enable adress sanitizer
option(ENABLE_ASAN "Enable Address Sanitizer" OFF)
if (ENABLE_ASAN)
    add_compile_options(-fsanitize=address)
    add_link_options(-fsanitize=address)
endif ()

# Dear ImGui
set(IMGUI_DIR /mnt5/tmp/src/imgui)
include_directories(${IMGUI_DIR} ${IMGUI_DIR}/backends ..)
#
## ImPlot
#set(IMPLOT_DIR ${DEP_DIR}/implot)
#include_directories(${IMPLOT_DIR})

# Ryzen
set(RYZEN_DIR /home/martin/src/ryzen_monitor/src)
include_directories(${RYZEN_DIR} ${RYZEN_DIR}/lib)

# Libraries
find_package(OpenGL REQUIRED)
set(LIBRARIES "OpenGL")

find_package(SDL2 REQUIRED CONFIG REQUIRED COMPONENTS SDL2)

set(sources
        src/main.cpp
        src/CpuAffinityManagerBase.cpp
        src/CpuAffinityManagerWithGui.cpp
        src/DiagramBase.cpp
        src/DiagramWithGui.cpp
        src/FancyWindow.cpp
)
# the files  in sources can be merged into one for a unity build
# place them into the same UNITY_GROUP
set_source_files_properties(${sources} PROPERTIES UNITY_GROUP "Sources")

set(imgui_sources_backends
        ${IMGUI_DIR}/backends/imgui_impl_sdl2.cpp
        ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp

)
#set_source_files_properties(${imgui_sources_backends} PROPERTIES UNITY_GROUP "ImGui_backends")

set(imgui_sources_separate
        ${IMGUI_DIR}/imgui.cpp
        #${IMGUI_DIR}/imgui_demo.cpp

        ${IMGUI_DIR}/imgui_draw.cpp

        ${IMGUI_DIR}/imgui_widgets.cpp
        ${IMGUI_DIR}/imgui_tables.cpp
)
set_source_files_properties(${imgui_sources_separate} PROPERTIES UNITY_GROUP "ImPlotImGui")
#
#set(implot_sources
#        ${IMPLOT_DIR}/implot.cpp
#        ${IMPLOT_DIR}/implot_items.cpp
#        #${IMPLOT_DIR}/implot_demo.cpp
#)
#set_source_files_properties(${implot_sources} PROPERTIES UNITY_GROUP "ImPlotImGui")

set(ryzen_sources
        ${RYZEN_DIR}/lib/libsmu.c
        ${RYZEN_DIR}/pm_tables.c
        ${RYZEN_DIR}/readinfo.c
        ${RYZEN_DIR}/ryzen_monitor.c
)
set_source_files_properties(${ryzen_sources} PROPERTIES UNITY_GROUP "Ryzen")
#file(GLOB sources src/*.cpp)

include_directories(/mnt5/tmp/src/popl/include/)

add_executable(${PROJECT_NAME}
        ${sources}
        ${imgui_sources_backends}
        ${imgui_sources_separate}
        ${implot_sources}
        ${ryzen_sources}
)

target_link_libraries(${PROJECT_NAME} ${LIBRARIES} SDL2::SDL2)
target_compile_definitions(${PROJECT_NAME} PUBLIC -DImTextureID=ImU64)

