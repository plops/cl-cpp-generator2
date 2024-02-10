cmake_minimum_required(VERSION 3.20)
project(daxa_test CXX)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 20)

set(GLFW_DIR ${DEP_DIR}/glfw) # Set this to point to an up-to-date GLFW repo
option(GLFW_BUILD_EXAMPLES "Build the GLFW example programs" OFF)
option(GLFW_BUILD_TESTS "Build the GLFW test programs" OFF)
option(GLFW_BUILD_DOCS "Build the GLFW documentation" OFF)
option(GLFW_INSTALL "Generate installation target" OFF)
option(GLFW_DOCUMENT_INTERNALS "Include internals in documentation" OFF)
add_subdirectory(${GLFW_DIR} binary_dir EXCLUDE_FROM_ALL)
include_directories(${GLFW_DIR}/include)



#find_package(OpenGL REQUIRED)
#set(LIBRARIES "glfw;OpenGL")

include_directories(${GLFW_DIR}/deps)

set(sources 
        src/main.cpp
        
)
