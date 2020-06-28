cmake_minimum_required(VERSION 3.16)
project(source)

set(CMAKE_CXX_STANDARD 17)

include_directories(.)
# add_definitions(-DNVRTC_GET_TYPE_NAME=1)
# link_directories(/media/sdb4/cuda/11.0.1/lib)

add_executable(source
globals.h
utils.h
vis_00_main.cpp
vis_00_main.hpp
)
# target_include_directories(source PRIVATE /media/sdb4/cuda/11.0.1/include)
# target_include_directories(source PRIVATE /home/martin/stage/cl-cpp-generator2/example/22_lmdb/source/lmdbpp/)
target_link_libraries(source lmdb)
