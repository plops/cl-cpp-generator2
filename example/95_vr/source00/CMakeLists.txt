cmake_minimum_required( VERSION 3.0 FATAL_ERROR )
project( main LANGUAGES CXX )
set( CMAKE_C_COMPILER /home/martin/quest2/ndk/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang )
set( CMAKE_CXX_COMPILER /home/martin/quest2/ndk/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang++ )
set( CMAKE_VERBOSE_MAKEFILE ON )
set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -ggdb -O0   -Wall -Wextra -std=c++20  ")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ggdb -O0   -Wall -Wextra -std=c++20  ")
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ggdb -O0   -Wall -Wextra -std=c++20  ")
set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -ggdb -O0  -Wall -Wextra -std=c++20 ")
set( SRCS /home/martin/stage/cl-cpp-generator2/example/95_vr/source00/App.cpp
/home/martin/stage/cl-cpp-generator2/example/95_vr/source00/AttribPointer.cpp
/home/martin/stage/cl-cpp-generator2/example/95_vr/source00/Egl.cpp
/home/martin/stage/cl-cpp-generator2/example/95_vr/source00/Geometry.cpp
/home/martin/stage/cl-cpp-generator2/example/95_vr/source00/Program.cpp
/home/martin/stage/cl-cpp-generator2/example/95_vr/source00/Renderer.cpp
/home/martin/stage/cl-cpp-generator2/example/95_vr/source00/Vertex.cpp
/home/martin/stage/cl-cpp-generator2/example/95_vr/source00/main.cpp
/home/martin/stage/cl-cpp-generator2/example/95_vr/source00/android_native_app_glue.c )
add_library (main SHARED ${SRCS} )
set_property( TARGET main PROPERTY CXX_STANDARD 20 )
target_include_directories( main PRIVATE
/home/martin/quest2/ndk/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/include/
/home/martin/quest2/ovr/VrApi/Include
)
add_library( vrapi SHARED IMPORTED )
set_target_properties( vrapi PROPERTIES IMPORTED_LOCATION /home/martin/quest2/ovr/VrApi/Libs/Android/arm64-v8a/Debug/libvrapi.so
 )
target_link_libraries( main PRIVATE android log vrapi )
