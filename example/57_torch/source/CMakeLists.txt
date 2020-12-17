cmake_minimum_required( VERSION 3.0 FATAL_ERROR )
project( mytest LANGUAGES CXX )
set( CMAKE_VERBOSE_MAKEFILE ON )
find_package( Torch REQUIRED )
set( SRCS vis_00_base.cpp
vis_01_demangle.cpp )
add_executable( mytest ${SRCS} )
target_link_libraries( mytest ${TORCH_LIBRARIES} )
set_property( TARGET mytest PROPERTY CXX_STANDARD 14 )
