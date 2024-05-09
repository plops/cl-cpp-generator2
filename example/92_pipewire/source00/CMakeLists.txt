cmake_minimum_required( VERSION 3.0 FATAL_ERROR )
project( mytest LANGUAGES CXX )
set( CMAKE_VERBOSE_MAKEFILE ON )
set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -ggdb -O0   -Wall -Wextra  ")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb -O0   -Wall -Wextra  ")
set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -ggdb -O0  -Wall -Wextra ")
set( SRCS /home/martin/stage/cl-cpp-generator2/example/92_pipewire/source00/main.cpp )
add_executable( mytest ${SRCS} )
set_property( TARGET mytest PROPERTY CXX_STANDARD 20 )
target_link_options( mytest PRIVATE -static-libgcc -static-libstdc++   )
find_package( PkgConfig REQUIRED )
pkg_check_modules( spdlog REQUIRED spdlog )
pkg_check_modules( pipewire REQUIRED libpipewire-0.3 )
target_include_directories( mytest PRIVATE
/usr/local/include/
/home/martin/src/popl/include/
/usr/include/pipewire-0.3
/usr/include/spa-0.2
/usr/include/glib-2.0
/usr/lib64/glib-2.0/include
/usr/include/sysprof-4  )
target_link_libraries( mytest PRIVATE spdlog pthread glib-2.0 gio-2.0 pipewire-0.3 )
