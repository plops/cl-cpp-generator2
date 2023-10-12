# CMake version
cmake_minimum_required(VERSION 3.10)

# Project name
project(imgui_sdl2_grpc_viewer)

# C++ Standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)

# set(CMAKE_CXX_COMPILER clang++)

option(ENABLE_MOLD "Enable Mold as Linker" ON)
if (ENABLE_MOLD)
    add_link_options(-fuse-ld=mold)
endif ()


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
            -O0 #3
            #-ffunction-sections -fdata-sections # don't use together with flto
    )
    # set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE) # flto
    #add_link_options(-Wl,-gc-sections,--print-gc-sections,--print-memory-usage,-Map=${PROJECT_BINARY_DIR}/${PROJECT_NAME}.map)
endif ()

set(OpenGL_GL_PREFERENCE
        GLVND
        # LEGACY
)

# Add SDL2
find_package(SDL2 REQUIRED)

# Add OpenGL
find_package(OpenGL REQUIRED)

find_package(protobuf CONFIG REQUIRED)
find_package(gRPC CONFIG REQUIRED)

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
        #-Wpadded
)


# gRPC service definition
set(PROTO_FILES
    proto/gl.proto
)

add_library(glproto ${PROTO_FILES})
target_link_libraries(glproto
    PUBLIC
        protobuf::libprotobuf
        gRPC::grpc
        gRPC::grpc++
)
target_include_directories(glproto PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

# Compile protobuf and grpc files in myproto target to cpp
get_target_property(grpc_cpp_plugin_location gRPC::grpc_cpp_plugin LOCATION)
protobuf_generate(TARGET glproto LANGUAGE cpp)
protobuf_generate(TARGET glproto LANGUAGE grpc GENERATE_EXTENSIONS .grpc.pb.h .grpc.pb.cc PLUGIN "protoc-gen-grpc=${grpc_cpp_plugin_location}")

add_library(imgui
        ${CMAKE_SOURCE_DIR}/imgui/imgui.cpp
        ${CMAKE_SOURCE_DIR}/imgui/imgui_demo.cpp
        ${CMAKE_SOURCE_DIR}/imgui/imgui_draw.cpp
        ${CMAKE_SOURCE_DIR}/imgui/backends/imgui_impl_sdl2.cpp
        ${CMAKE_SOURCE_DIR}/imgui/backends/imgui_impl_opengl3.cpp
        ${CMAKE_SOURCE_DIR}/imgui/imgui_widgets.cpp
        ${CMAKE_SOURCE_DIR}/imgui/imgui_tables.cpp
)

target_include_directories(
        imgui
        PRIVATE
        ${SDL2_INCLUDE_DIRS}
        ${OPENGL_INCLUDE_DIR}
        ${CMAKE_SOURCE_DIR}/imgui
        ${CMAKE_SOURCE_DIR}/imgui/backends
)

target_include_directories(
        ${PROJECT_NAME}
        PRIVATE
        ${SDL2_INCLUDE_DIRS}
        ${OPENGL_INCLUDE_DIR}
        ${CMAKE_SOURCE_DIR}/imgui
        ${CMAKE_SOURCE_DIR}/imgui/backends
)

# Linking libraries
target_link_libraries(${PROJECT_NAME}
        ${SDL2_LIBRARIES}
        ${OPENGL_LIBRARIES}
        imgui
	glproto
)
