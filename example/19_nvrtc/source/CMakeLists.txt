cmake_minimum_required(VERSION 3.16)
project(source)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)

add_executable(source
        bla.cu
        globals.h
        utils.h
        vis_00_main.cpp
        vis_00_main.hpp
        vis_01_rtc.cpp
        vis_01_rtc.hpp
        vis_02_cu_device.cpp
        vis_02_cu_device.hpp
        vis_03_cu_program.cpp
        vis_03_cu_program.hpp
        vis_04_cu_module.cpp
        vis_04_cu_module.hpp
        vis_05_cu_compilation_options.cpp
        vis_05_cu_compilation_options.hpp)
target_include_directories(source PRIVATE /media/sdb4/cuda/11.0.1/include)
