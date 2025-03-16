cmake_minimum_required(VERSION 3.13)
project(mytest LANGUAGES CXX)
set(CMAKE_CXX_COMPILER g++)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -ggdb -O0   -Wall -Wextra -ffile-prefix-map=/home/martin/stage/cl-cpp-generator2/example/85_glbinding_av/source01/= ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb -O0   -Wall -Wextra -ffile-prefix-map=/home/martin/stage/cl-cpp-generator2/example/85_glbinding_av/source01/= ")
set(CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -ggdb -O0  -Wall -Wextra ")
set(SRCS /home/martin/stage/cl-cpp-generator2/example/85_glbinding_av/source01/GlfwWindow.cpp
        /home/martin/stage/cl-cpp-generator2/example/85_glbinding_av/source01/ImguiHandler.cpp
        /home/martin/stage/cl-cpp-generator2/example/85_glbinding_av/source01/Texture.cpp
        /home/martin/stage/cl-cpp-generator2/example/85_glbinding_av/source01/Video.cpp
        /home/martin/stage/cl-cpp-generator2/example/85_glbinding_av/source01/main.cpp
        /home/martin/src/imgui/backends/imgui_impl_opengl3.cpp
        /home/martin/src/imgui/backends/imgui_impl_glfw.cpp
        /home/martin/src/imgui/imgui.cpp
        /home/martin/src/imgui/imgui_demo.cpp
        /home/martin/src/imgui/imgui_draw.cpp
        /home/martin/src/imgui/imgui_tables.cpp
        /home/martin/src/imgui/imgui_widgets.cpp)
add_executable(mytest ${SRCS})
target_include_directories(mytest PRIVATE
        /usr/include
        /home/martin/src/imgui/
        /usr/local/include
        /home/martin/src/popl/include/
)
target_compile_features(mytest PUBLIC cxx_std_20)
target_link_options(mytest PRIVATE -static-libgcc -static-libstdc++)
#find_package( glbinding )
find_package(glfw3 REQUIRED)
add_library(libavcpp_static STATIC IMPORTED)
set_target_properties(libavcpp_static PROPERTIES IMPORTED_LOCATION /usr/local/lib/libavcpp.a)
#find_package(PkgConfig REQUIRED)
#pkg_check_modules(spdlog  )
target_link_libraries(mytest PRIVATE #glbinding::glbinding
        glfw3 libavcpp_static avutil avdevice avfilter avcodec avformat swscale postproc swresample
        #spdlog
)
