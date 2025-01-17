cmake_minimum_required(VERSION 3.6)
project(shape)
set(CMAKE_CXX_STANDARD 26)
add_executable(
        ${PROJECT_NAME}
        Circle.cpp
        Shape.cpp
        main.cpp
)