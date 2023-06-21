cmake_minimum_required(VERSION 3.10)
project(LMS_Equalizer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(LMS_Equalizer main.cpp)
add_executable(LMS_Equalizer_c q.c)
target_link_libraries(LMS_Equalizer_c m c)