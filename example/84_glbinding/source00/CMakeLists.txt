cmake_minimum_required( VERSION 3.13 )
project( mytest LANGUAGES CXX )
set( CMAKE_CXX_COMPILER g++ )
set( CMAKE_VERBOSE_MAKEFILE ON )
set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -ggdb -O0 -fno-omit-frame-pointer -fsanitize=address -fsanitize-address-use-after-return=always -fsanitize-address-use-after-scope  -Wall -Wextra -Wcast-align -Wcast-qual -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2 -Winit-self  -Wmissing-declarations -Wmissing-include-dirs -Wold-style-cast -Woverloaded-virtual -Wredundant-decls -Wshadow -Wsign-conversion -Wswitch-default -Wundef -Werror -Wno-unused ")
set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -ggdb -O0  -Wall -Wextra -Wcast-align -Wcast-qual -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2 -Winit-self  -Wmissing-declarations -Wmissing-include-dirs -Wold-style-cast -Woverloaded-virtual -Wredundant-decls -Wshadow -Wsign-conversion -Wswitch-default -Wundef -Werror -Wno-unused ")
set( SRCS /home/martin/stage/cl-cpp-generator2/example/84_glbinding/source00/main.cpp
/home/martin/src/imgui/backends/imgui_impl_opengl3.cpp
/home/martin/src/imgui/backends/imgui_impl_glfw.cpp
/home/martin/src/imgui/imgui.cpp
/home/martin/src/imgui/imgui_demo.cpp
/home/martin/src/imgui/imgui_draw.cpp
/home/martin/src/imgui/imgui_tables.cpp
/home/martin/src/imgui/imgui_widgets.cpp )
add_executable( mytest ${SRCS} )
target_include_directories( mytest PRIVATE
/home/martin/src/imgui/
 )
target_compile_features( mytest PUBLIC cxx_std_20 )
target_link_options( mytest PRIVATE -static-libgcc -static-libstdc++   )
find_package( glbinding REQUIRED )
find_package( glfw3 REQUIRED )
target_link_libraries( mytest PRIVATE glbinding::glbinding glfw3 )
