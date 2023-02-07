cmake_minimum_required(VERSION 3.10)
set(PROJECT_NAME white_noise_audio)
project(${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 17)



find_package(PkgConfig REQUIRED)
pkg_check_modules(PIPEWIRE REQUIRED pipewire-0.3)

include_directories(${PIPEWIRE_INCLUDE_DIRS})
link_directories(${PIPEWIRE_LIBRARY_DIRS})

add_executable(${PROJECT_NAME} ${PROJECT_NAME}.cpp)
target_link_libraries(${PROJECT_NAME} ${PIPEWIRE_LIBRARIES})
