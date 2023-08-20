cmake_minimum_required(VERSION 3.18)

set(TARGET_NAME example_target)
set(PROJECT_NAME example_project)

project(${PROJECT_NAME} VERSION 0.1.0 LANGUAGES CXX)

# Point to the directory containing mousetrap's CMake config files
list(APPEND CMAKE_PREFIX_PATH "/home/martin/moustrap/lib64/cmake/mousetrap")

find_package(mousetrap REQUIRED)

add_executable(${TARGET_NAME} src/main.cpp)

target_link_libraries(${TARGET_NAME} PRIVATE
    ${MOUSETRAP_LIBRARIES}
)

message(STATUS "MOUSETRAP_INCLUDE_DIRECTORIES = ${MOUSETRAP_INCLUDE_DIRECTORIES}")


target_include_directories(${TARGET_NAME} PRIVATE
    #${MOUSETRAP_INCLUDE_DIRECTORIES}
    /home/martin/moustrap/include/
)

target_compile_features(${TARGET_NAME} PUBLIC
    cxx_std_17
)
