cmake_minimum_required(VERSION 3.16)
project(source)

set(CMAKE_CXX_STANDARD 17)

include_directories(.)
add_definitions(-DNVRTC_GET_TYPE_NAME=1)
link_directories(/media/sdb4/cuda/11.0.1/lib)

add_executable(source
globals.h
utils.h
vis_00_main.cpp
vis_00_main.hpp
yacx/src/Init.cpp
yacx/src/KernelArg.cpp
yacx/src/Colors.cpp
yacx/src/Devices.cpp
yacx/src/cexecutor/LibaryLoader.cpp
yacx/src/cexecutor/CProgram.cpp
yacx/src/util.cpp
yacx/src/Headers.cpp
yacx/src/Logger.cpp
yacx/src/Exception.cpp
yacx/src/KernelTime.cpp
yacx/src/Program.cpp
yacx/src/Device.cpp
yacx/src/Options.cpp
yacx/src/Source.cpp
yacx/src/KernelArgs.cpp
yacx/src/Kernel.cpp
)
target_include_directories(source PRIVATE /media/sdb4/cuda/11.0.1/include)
target_include_directories(source PRIVATE /home/martin/stage/cl-cpp-generator2/example/21_nvrtc_yacx/source/yacx/include)
target_link_libraries(source cuda cudart nvrtc dl)
