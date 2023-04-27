cmake_minimum_required(VERSION 2.9)
project(music)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


find_package(PkgConfig REQUIRED)
pkg_check_modules(PORTAUDIO REQUIRED portaudio-2.0)

include_directories(${PORTAUDIO_INCLUDE_DIRS})
link_directories(${PORTAUDIO_LIBRARY_DIRS})


add_executable(music
 main.cpp
 WavetableOscillator.cpp
 EnvelopeGenerator.cpp
 )
 
target_link_libraries(music
 ${PORTAUDIO_LIBRARIES})
