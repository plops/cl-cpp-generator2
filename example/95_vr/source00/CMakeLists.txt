cmake_minimum_required( VERSION 3.0 FATAL_ERROR )
project( mytest LANGUAGES CXX )
set( CMAKE_CXX_COMPILER aarch64-linux-android29-clang++ )
set( CMAKE_VERBOSE_MAKEFILE ON )
set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -ggdb -O0   -Wall -Wextra -std=c++20 -march=armv8-a   ")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb -O0   -Wall -Wextra -std=c++20 -march=armv8-a   ")
set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -ggdb -O0  -Wall -Wextra -std=c++20 -march=armv8-a  ")
set( SRCS /home/martin/stage/cl-cpp-generator2/example/95_vr/source00/App.cpp
/home/martin/stage/cl-cpp-generator2/example/95_vr/source00/AttribPointer.cpp
/home/martin/stage/cl-cpp-generator2/example/95_vr/source00/Cube.cpp
/home/martin/stage/cl-cpp-generator2/example/95_vr/source00/Egl.cpp
/home/martin/stage/cl-cpp-generator2/example/95_vr/source00/Framebuffer.cpp
/home/martin/stage/cl-cpp-generator2/example/95_vr/source00/Geometry.cpp
/home/martin/stage/cl-cpp-generator2/example/95_vr/source00/Program.cpp
/home/martin/stage/cl-cpp-generator2/example/95_vr/source00/Renderer.cpp
/home/martin/stage/cl-cpp-generator2/example/95_vr/source00/Vertex.cpp
/home/martin/stage/cl-cpp-generator2/example/95_vr/source00/main.cpp )
add_library (mytest SHARED ${SRCS} )
set_property( TARGET mytest PROPERTY CXX_STANDARD 20 )
target_include_directories( mytest PRIVATE

)
target_link_libraries( mytest PRIVATE android log vrapi )
