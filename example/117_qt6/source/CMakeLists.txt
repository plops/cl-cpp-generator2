
cmake_minimum_required(VERSION 3.16)

project(my_project)

set(CMAKE_CXX_STANDARD 20)

find_package(Qt6 COMPONENTS Core Gui Widgets Svg Charts REQUIRED)

add_executable(my_project main.cpp)

target_link_libraries(my_project PRIVATE Qt6::Core Qt6::Gui Qt6::Widgets Qt6::Svg Qt6::Charts)
