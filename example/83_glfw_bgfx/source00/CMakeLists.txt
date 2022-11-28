cmake_minimum_required( VERSION 3.13 )
project( mytest LANGUAGES CXX )
set( CMAKE_CXX_COMPILER clang++ )
set( CMAKE_VERBOSE_MAKEFILE ON )
set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -ggdb -O0 -fno-omit-frame-pointer -fsanitize=address -fsanitize-address-use-after-return=always -fsanitize-address-use-after-scope  -Wall -Wextra -Wcast-align -Wcast-qual -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2 -Winit-self  -Wmissing-declarations -Wmissing-include-dirs -Wold-style-cast -Woverloaded-virtual -Wredundant-decls -Wshadow -Wsign-conversion -Wswitch-default -Wundef -Werror -Wno-unused ")
set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -ggdb -O0  -Wall -Wextra -Wcast-align -Wcast-qual -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2 -Winit-self  -Wmissing-declarations -Wmissing-include-dirs -Wold-style-cast -Woverloaded-virtual -Wredundant-decls -Wshadow -Wsign-conversion -Wswitch-default -Wundef -Werror -Wno-unused ")
set( SRCS /home/martin/stage/cl-cpp-generator2/example/83_glfw_bgfx/source00/PosColorVertex.cpp
/home/martin/stage/cl-cpp-generator2/example/83_glfw_bgfx/source00/main.cpp
/home/martin/src/bgfx/examples/common/imgui/imgui.cpp
/home/martin/src/bgfx/3rdparty/dear-imgui/imgui.cpp
/home/martin/src/bgfx/3rdparty/dear-imgui/imgui_demo.cpp
/home/martin/src/bgfx/3rdparty/dear-imgui/imgui_draw.cpp
/home/martin/src/bgfx/3rdparty/dear-imgui/imgui_tables.cpp
/home/martin/src/bgfx/3rdparty/dear-imgui/imgui_widgets.cpp )
add_executable( mytest ${SRCS} )
target_include_directories( mytest PRIVATE
/home/martin/src/bgfx/include/
/home/martin/src/bx/include/
/home/martin/src/bimg/include/
/home/martin/src/bgfx/examples/common
/home/martin/src/bgfx/3rdparty/
/home/martin/src/entt/src/ )
target_compile_features( mytest PUBLIC cxx_std_20 )
add_library( bgfx-shared SHARED IMPORTED )
set_target_properties( bgfx-shared PROPERTIES
IMPORTED_LOCATION "/home/martin/src/bgfx/.build/linux64_gcc/bin/libbgfx-shared-libRelease.so"
INTERFACE_INCLUDE_DIRECTORIES "/home/martin/src/bgfx/include" )
add_definitions( -DBX_CONFIG_DEBUG )
target_link_libraries( mytest PRIVATE bgfx-shared GL X11 glfw dl pthread rt )
