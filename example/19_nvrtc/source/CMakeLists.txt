cmake_minimum_required(VERSION 3.16)
project(source)

set(CMAKE_CXX_STANDARD 17)

include_directories(.)
add_definitions(-DNVRTC_GET_TYPE_NAME=1)
add_executable(source
bla.cu
globals.h
utils.h
vis_00_main.cpp
vis_00_main.hpp
vis_01_cu_A_rtc_code.cpp
vis_01_cu_A_rtc_code.hpp
vis_02_cu_A_device.cpp
vis_02_cu_A_device.hpp
vis_03_cu_A_context.cpp
vis_03_cu_A_context.hpp
vis_04_cu_A_rtc_compilation_options.cpp
vis_04_cu_A_rtc_compilation_options.hpp
vis_05_cu_A_rtc_header.cpp
vis_05_cu_A_rtc_header.hpp
vis_06_cu_A_rtc_program.cpp
vis_06_cu_A_rtc_program.hpp
vis_07_cu_A_rtc_kernel.cpp
vis_07_cu_A_rtc_kernel.hpp
vis_08_cu_A_rtc_module.cpp
vis_08_cu_A_rtc_module.hpp)
target_include_directories(source PRIVATE /media/sdb4/cuda/11.0.1/include)
