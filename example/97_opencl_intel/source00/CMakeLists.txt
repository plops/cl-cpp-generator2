cmake_minimum_required( VERSION 3.0 FATAL_ERROR )
project( mytest LANGUAGES CXX )
set( CMAKE_VERBOSE_MAKEFILE ON )
set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -ggdb -O0   -Wall -Wextra  ")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb -O0   -Wall -Wextra  ")
set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -ggdb -O0  -Wall -Wextra ")
set( SRCS /home/martin/stage/cl-cpp-generator2/example/97_opencl_intel/source00/main.cpp )
add_executable( mytest ${SRCS} )
set_property( TARGET mytest PROPERTY CXX_STANDARD 20 )
target_link_options( mytest PRIVATE -static-libgcc -static-libstdc++   )
find_package( PkgConfig REQUIRED )
pkg_check_modules( spdlog REQUIRED spdlog )
pkg_check_modules( OpenCL REQUIRED OpenCL )
target_include_directories( mytest PRIVATE
/usr/local/include/
/home/martin/src/popl/include/
)
target_link_libraries( mytest PRIVATE spdlog OpenCL )
target_precompile_headers( mytest PRIVATE fatheader.hpp )
