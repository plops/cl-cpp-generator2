cmake_minimum_required(VERSION 3.17)
project(imgui_grpc)
set(CMAKE_CXX_STANDARD 20)

set(
        SOURCE_FILES
        src/main.cpp
	src/glgui.grpc.pb.cc
	src/glgui.pb.cc
)

find_package(gRPC CONFIG REQUIRED)

include(FetchContent)
FetchContent_Declare(
        imgui-glfw-glad-glm
        GIT_REPOSITORY https://github.com/cmmw/imgui-glfw-glad-glm.git
        GIT_TAG v4.0.1
)

FetchContent_MakeAvailable(imgui-glfw-glad-glm)

# set compiler flags
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(
            -g -g3 -ggdb -gdwarf-4
            -Wall -Wextra -Wconversion -Wdouble-promotion
            -Wno-unused-parameter -Wno-unused-function -Wno-sign-conversion
            -Wvla -Wframe-larger-than=5000 -Wstack-usage=10000
            -Wshadow
            -fvisibility=hidden
            -fno-strict-overflow -Wno-strict-overflow
            -fsanitize=undefined
    )
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=undefined")
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
	gRPC::grpc++
)
 
