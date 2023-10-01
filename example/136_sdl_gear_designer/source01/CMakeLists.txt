# CMake version
cmake_minimum_required(VERSION 3.10)

# Project name
project(imgui_sdl2_bullet_gears_designer)

# C++ Standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)


# AddressSanitizer
option(ENABLE_ASAN "Enable AddressSanitizer" OFF)
if (ENABLE_ASAN)
    add_compile_options(-fsanitize=address)
    add_link_options(-fsanitize=address)
endif ()

# UndefinedBehaviorSanitizer
option(ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)
if (ENABLE_UBSAN)
    add_compile_options(-fsanitize=undefined)
    add_link_options(-fsanitize=undefined)
endif ()

# ThreadSanitizer
option(ENABLE_TSAN "Enable ThreadSanitizer" OFF)
if (ENABLE_TSAN)
    add_compile_options(-fsanitize=thread)
    add_link_options(-fsanitize=thread)
endif ()

# LeakSanitizer
option(ENABLE_LSAN "Enable LeakSanitizer" OFF)
if (ENABLE_LSAN)
    add_compile_options(-fsanitize=leak)
    add_link_options(-fsanitize=leak)
endif ()


# set compiler flags
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(
            -g -g3 -ggdb -gdwarf-4
    )
else ()
    add_compile_options(
            -O3
            #-ffunction-sections -fdata-sections # don't use together with flto
    )
    # flto
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    #add_link_options(-Wl,-gc-sections,--print-gc-sections,--print-memory-usage,-Map=${PROJECT_BINARY_DIR}/${PROJECT_NAME}.map)
endif ()

set(OpenGL_GL_PREFERENCE
        GLVND
        # LEGACY
)

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
)

target_compile_options(
        ${PROJECT_NAME} PRIVATE
        -Wall -Wextra -Wconversion -Wdouble-promotion
        -Wno-unused-parameter -Wno-unused-function -Wno-sign-conversion
        -Wvla -Wframe-larger-than=5000 -Wstack-usage=10000
        -Wshadow -Wformat=2 -fno-common -fstack-usage
        -fvisibility=hidden
        -fno-strict-overflow -Wno-strict-overflow
        -Wpadded
)


add_library(imgui
        ${CMAKE_SOURCE_DIR}/imgui/imgui.cpp
        ${CMAKE_SOURCE_DIR}/imgui/imgui_demo.cpp
        ${CMAKE_SOURCE_DIR}/imgui/imgui_draw.cpp
        ${CMAKE_SOURCE_DIR}/imgui/backends/imgui_impl_sdl2.cpp
        ${CMAKE_SOURCE_DIR}/imgui/backends/imgui_impl_opengl3.cpp
        ${CMAKE_SOURCE_DIR}/imgui/imgui_widgets.cpp
        ${CMAKE_SOURCE_DIR}/imgui/imgui_tables.cpp
)
# Linking libraries
target_link_libraries(${PROJECT_NAME}
        ${SDL2_LIBRARIES}
        ${OPENGL_LIBRARIES}
        imgui
)
