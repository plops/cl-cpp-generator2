cmake_minimum_required( VERSION 3.4 )
project( mytest )
find_package( pybind11 REQUIRED )
set( SRCS /home/martin/stage/cl-cpp-generator2/example/49_pybind/source/vis_00_base.cpp
add_executable( mytest ${SRCS} )
target_link_libraries( mytest PRIVATE pybind11::embed )
