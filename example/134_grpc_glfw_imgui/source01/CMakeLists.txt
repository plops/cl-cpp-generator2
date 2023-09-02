cmake_minimum_required(VERSION 3.17)
project(imgui_grpc)
set(CMAKE_CXX_STANDARD 20)

set(
        SOURCE_FILES
        src/main.cpp
        src/glgui.grpc.pb.cc
        src/glgui.pb.cc
)

#find_package(gRPC CONFIG )
#find_package(protobuf CONFIG )

if (gRPC_FOUND) #AND protobuf_FOUND)

    message(STATUS "Using gRPC ${gRPC_VERSION}")
    #message(STATUS "Using protobuf ${protobuf_VERSION}")
else ()
    # Fallback to pkg-config
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(GRPC REQUIRED grpc++)
    pkg_check_modules(PROTOBUF REQUIRED protobuf)

    message(STATUS "Using gRPC via pkg-config")
endif ()

include(FetchContent)
FetchContent_Declare(
        imgui-glfw-glad-glm
        GIT_REPOSITORY https://github.com/cmmw/imgui-glfw-glad-glm.git
        GIT_TAG v4.0.1
)

FetchContent_MakeAvailable(imgui-glfw-glad-glm)

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


# https://interrupt.memfault.com/blog/best-and-worst-gcc-clang-compiler-flags

add_compile_options(
        -Wall -Wextra -Wconversion -Wdouble-promotion
        -Wno-unused-parameter -Wno-unused-function -Wno-sign-conversion
        -Wvla -Wframe-larger-than=5000 -Wstack-usage=10000
        -Wshadow -Wformat=2 -fno-common -fstack-usage
        -fvisibility=hidden
        -fno-strict-overflow -Wno-strict-overflow
)

# set compiler flags
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(
            -g -g3 -ggdb -gdwarf-4
    )
else ()
    add_compile_options(
            -Ofast
            -ffunction-sections -fdata-sections
            -Wpadded
    )
    add_link_options(-Wl,-gc-sections,--print-gc-sections,--print-memory-usage,-Map=${PROJECT_BINARY_DIR}/${PROJECT_NAME}.map)
endif ()

include_directories(${CMAKE_SOURCE_DIR}/src)
add_executable(
        ${PROJECT_NAME}
        ${SOURCE_FILES}
)

# It is sufficient to only link glm and ImGui since the latter one already contains Glad and GLFW
target_link_libraries(
        ${PROJECT_NAME}
        imgui
        glm

)
if (gRPC_FOUND)
    target_link_libraries(${PROJECT_NAME} gRPC::grpc++)
else ()
    # Include directories
    target_include_directories(${PROJECT_NAME} PUBLIC ${GRPC_INCLUDE_DIRS} ${PROTOBUF_INCLUDE_DIRS})

    # Compiler options
    target_compile_options(${PROJECT_NAME} PUBLIC ${GRPC_CFLAGS_OTHER})

    # Link libraries
    target_link_libraries(${PROJECT_NAME}
            ${GRPC_LIBRARIES}
            ${PROTOBUF_LIBRARIES}
    )

endif ()

# https://github.com/faaxm/exmpl-cmake-grpc/blob/master/proto/CMakeLists.txt
#set(PROTO_FILES
#    src/glgui.proto
#)
#get_target_property(grpc_cpp_plugin_location gRPC::grpc_cpp_plugin LOCATION)
#protobuf_generate(TARGET myproto LANGUAGE cpp)
#protobuf_generate(TARGET myproto LANGUAGE grpc GENERATE_EXTENSIONS .grpc.pb.h .grpc.pb.cc PLUGIN "protoc-gen-grpc=${grpc_cpp_plugin_location}")
