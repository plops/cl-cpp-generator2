cmake_minimum_required(VERSION 3.6)
project(dogcat1)
set(CMAKE_CXX_STANDARD 23)
add_executable(
        ${PROJECT_NAME}
        main.cpp
)
add_executable(
        dogcat2
        main2.cpp
)
add_executable(
        dogcat3
        main3.cpp
)