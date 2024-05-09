cmake_minimum_required( VERSION 3.0 FATAL_ERROR )
project( mytest )
set( CMAKE_VERBOSE_MAKEFILE ON )
set( SRCS /home/martin/stage/cl-cpp-generator2/example/98_opencl_wr/source00/kernel.cpp
/home/martin/stage/cl-cpp-generator2/example/98_opencl_wr/source00/main.cpp )
add_executable( mytest ${SRCS} )
set_property( TARGET mytest PROPERTY CXX_STANDARD 20 )
target_link_options( mytest PRIVATE -static-libgcc -static-libstdc++   )
find_package( PkgConfig REQUIRED )
pkg_check_modules( spdlog REQUIRED spdlog )
pkg_check_modules( OpenCL REQUIRED OpenCL )
target_include_directories( mytest PRIVATE
/home/martin/stage/cl-cpp-generator2/example/98_opencl_wr/wrapper/
)
target_link_libraries( mytest PRIVATE spdlog OpenCL )
target_precompile_headers( mytest PRIVATE fatheader.hpp )
