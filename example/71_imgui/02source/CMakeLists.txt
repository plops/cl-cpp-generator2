cmake_minimum_required( VERSION 3.4 )
project( mytest LANGUAGES CXX )
set( CMAKE_CXX_COMPILER clang++ )
set( CMAKE_VERBOSE_MAKEFILE ON )
set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -ggdb -O0  -fno-omit-frame-pointer -fsanitize=address -fsanitize-address-use-after-return=always -fsanitize-address-use-after-scope  -Wall -Wextra -Wcast-align -Wcast-qual -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2 -Winit-self  -Wmissing-declarations -Wmissing-include-dirs  -Woverloaded-virtual -Wredundant-decls -Wshadow  -Wswitch-default -Wundef -Werror -Wno-unused ")
set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -ggdb -O0   -Wall -Wextra -Wcast-align -Wcast-qual -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2 -Winit-self  -Wmissing-declarations -Wmissing-include-dirs  -Woverloaded-virtual -Wredundant-decls -Wshadow  -Wswitch-default -Wundef -Werror -Wno-unused ")
set( SRCS /home/martin/stage/cl-cpp-generator2/example/71_imgui/02source/BoardProcessor.cpp
/home/martin/stage/cl-cpp-generator2/example/71_imgui/02source/Charuco.cpp
/home/martin/stage/cl-cpp-generator2/example/71_imgui/02source/GraphicsFramework.cpp
/home/martin/stage/cl-cpp-generator2/example/71_imgui/02source/MainWindow.cpp
/home/martin/stage/cl-cpp-generator2/example/71_imgui/02source/ProcessFrameEvent.cpp
/home/martin/stage/cl-cpp-generator2/example/71_imgui/02source/ProcessedFrameMessage.cpp
/home/martin/stage/cl-cpp-generator2/example/71_imgui/02source/main.cpp )
add_executable( mytest ${SRCS} )
target_compile_features( mytest PUBLIC cxx_std_17 )
find_package( imgui CONFIG REQUIRED )
find_package( implot CONFIG REQUIRED )
find_package( OpenCV REQUIRED core videoio imgproc aruco )
target_link_libraries( mytest PRIVATE imgui::imgui implot::implot ${OpenCV_LIBS} )
