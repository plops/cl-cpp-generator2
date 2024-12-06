cmake_minimum_required(VERSION 3.5)

project(showGrating)

set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)

set(GLFWPP_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE) # disable building glfwpp examples
add_subdirectory(ext/glfwpp)
set(OpenGL_GL_PREFERENCE GLVND)
find_package(OpenGL)
#find_package(GLEW)
# include_directories(/home/martin/eigen/include/eigen3)
#find_package(popl)
#find_package(Eigen3)

if (CMAKE_BUILD_TYPE STREQUAL Debug)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb3 -O0")
endif ()

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    if (NOT CMAKE_CXX_FLAGS MATCHES "-march" AND NOT CMAKE_CXX_FLAGS MATCHES "-arch" AND NOT CMAKE_OSX_ARCHITECTURES)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -mtune=native -ggdb3")
    endif ()
    if (NOT CMAKE_CXX_COMPILER_ID MATCHES Clang) # We are using clang-cl
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
    endif ()
endif ()

# Specify C++ standard (e.g., C++17)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Enable AddressSanitizer
option(ENABLE_ASAN "enable address sanitizer" OFF)
if (ENABLE_ASAN)
    add_compile_options(-fsanitize=address)
    add_link_options(-fsanitize=address)
endif ()


add_executable(${PROJECT_NAME} src/main.cpp)
#set_property(TARGET meanSimd PROPERTY CXX_STANDARD 23)

target_link_libraries(${PROJECT_NAME} PRIVATE GLFWPP OpenGL)
