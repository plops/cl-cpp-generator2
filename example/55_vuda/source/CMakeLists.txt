cmake_minimum_required( VERSION 3.4 )
project( mytest LANGUAGES CXX )
set( CMAKE_VERBOSE_MAKEFILE ON )
set( CMAKE_CXX_STANDARD 17 )
set( SRCS /home/martin/stage/cl-cpp-generator2/example/55_vuda/source/vis_00_base.cpp
/home/martin/stage/cl-cpp-generator2/example/55_vuda/source/vis_01_demangle.cpp )
add_executable( mytest ${SRCS} )
target_include_directories( mytest PUBLIC /home/martin/src/vuda/inc /home/martin/stage/cl-cpp-generator2/example/55_vuda/source/ )
target_link_libraries( mytest PRIVATE vulkan )
target_precompile_headers( mytest PRIVATE vis_00_base.hpp )
