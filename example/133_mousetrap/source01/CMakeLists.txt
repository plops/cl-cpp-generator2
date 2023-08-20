cmake_minimum_required(VERSION 3.18)

set(TARGET_NAME example_target)
set(PROJECT_NAME example_project)

project(${PROJECT_NAME} VERSION 0.1.0 LANGUAGES CXX)

find_package(mousetrap REQUIRED)

add_executable(${TARGET_NAME} main.cpp)

target_link_libraries(${TARGET_NAME} PRIVATE
    ${MOUSETRAP_LIBRARIES}
)

target_include_directories(${TARGET_NAME} PRIVATE
    ${MOUSETRAP_INCLUDE_DIRECTORIES}
)

target_compile_features(${TARGET_NAME} PUBLIC
    cxx_std_17
)
