cmake_minimum_required(VERSION 3.10)
set(PROJECT_NAME white_noise_audio)
project(${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 23)



find_package(PkgConfig REQUIRED)
pkg_check_modules(PIPEWIRE REQUIRED libpipewire-0.3)
find_package(fmt REQUIRED)

include_directories(${PIPEWIRE_INCLUDE_DIRS} ${fmt_INCLUDE_DIRS})
link_directories(${PIPEWIRE_LIBRARY_DIRS})

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} ${PIPEWIRE_LIBRARIES} fmt::fmt)
