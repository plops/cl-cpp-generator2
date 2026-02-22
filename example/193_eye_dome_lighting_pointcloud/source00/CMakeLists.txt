cmake_minimum_required(VERSION 3.10)
project(PointcloudEDL CXX C)

# Enforce Modern C++ Standards
set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# AddressSanitizer
option(ENABLE_UBLASAN "Enable AddressSanitizer" OFF)
if (ENABLE_UBLASAN)
    add_compile_options(-fsanitize=address -fsanitize=undefined -fsanitize=undefined)
    add_link_options(-fsanitize=address -fsanitize=undefined -fsanitize=undefined)
endif ()

# ThreadSanitizer
option(ENABLE_TSAN "Enable ThreadSanitizer" OFF)
if (ENABLE_TSAN)
    add_compile_options(-fsanitize=thread)
    add_link_options(-fsanitize=thread)
endif ()

if(NOT WIN32)
    set(CMAKE_PREFIX_PATH /home/martin/grpc/)
    # https://interrupt.memfault.com/blog/best-and-worst-gcc-clang-compiler-flags

    set(STRICT_WARNING_FLAGS
        -Wall -Wextra -Wconversion -Wdouble-promotion
        -Wno-unused-parameter -Wno-unused-function -Wno-sign-conversion
        -Wvla -Wframe-larger-than=5000 -Wstack-usage=10000
        -Wshadow -Wformat=2 -fno-common -fstack-usage
        -fvisibility=hidden
        -fno-strict-overflow -Wno-strict-overflow
	)

endif()


# set compiler flags
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(
            -g -g3 -ggdb -gdwarf-4
    )
else ()
    add_compile_options(
	    -march=native
	    -floop-nest-optimize -fgraphite-identity # polyhedral loop optimization
	    -floop-parallelize-all
	    -floop-interchange # switches inner and outer loop to improve caching
	    -floop-strip-mine # split loop into smaller loops to fit into cache
	    -floop-block # perform tiling to improve data locality
            -Ofast
            -ffunction-sections -fdata-sections
            -Wpadded
    )
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    add_link_options(-Wl,-gc-sections,--print-gc-sections,--print-memory-usage,-Map=${PROJECT_BINARY_DIR}/${PROJECT_NAME}.map)
endif ()




# Expand the home directory path to locate dependencies
set(SRC_DIR "$ENV{HOME}/src")

# 1. GLM (Header-only mathematics library)
add_library(glm INTERFACE)
target_include_directories(glm INTERFACE ${SRC_DIR}/glm)

# 2. GLAD (OpenGL loader generated via web interface, OpenGL 3.3 Core with texture object and depth texture)
add_library(glad STATIC glad/src/glad.c)
target_include_directories(glad PUBLIC glad/include)

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
add_executable(PointcloudEDL src/main.cpp)
target_compile_options(PointcloudEDL PRIVATE ${STRICT_WARNING_FLAGS})

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
