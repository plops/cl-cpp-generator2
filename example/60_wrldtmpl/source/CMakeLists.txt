cmake_minimum_required( VERSION 3.4 )
project( mytest LANGUAGES CXX )
set( CMAKE_CXX_COMPILER nvc++ )
set( CMAKE_CXX_FLAGS "-stdpar"  )
set( CMAKE_VERBOSE_MAKEFILE ON )
set( CMAKE_CXX_STANDARD 17 )
set( SRCS /home/martin/stage/cl-cpp-generator2/example/60_wrldtmpl/source/vis_00_base.cpp
/home/martin/stage/cl-cpp-generator2/example/60_wrldtmpl/source/vis_01_demangle.cpp
/home/martin/stage/cl-cpp-generator2/example/60_wrldtmpl/source/vis_02_surface95.cpp
/home/martin/stage/cl-cpp-generator2/example/60_wrldtmpl/source/vis_03_memory.cpp
/home/martin/stage/cl-cpp-generator2/example/60_wrldtmpl/source/vis_04_gl_texture.cpp
/home/martin/stage/cl-cpp-generator2/example/60_wrldtmpl/source/vis_05_gl_shader.cpp
/home/martin/stage/cl-cpp-generator2/example/60_wrldtmpl/source/vis_06_gl_helper.cpp
/home/martin/stage/cl-cpp-generator2/example/60_wrldtmpl/source/vis_07_job.cpp
/home/martin/stage/cl-cpp-generator2/example/60_wrldtmpl/source/vis_08_random.cpp
/home/martin/stage/cl-cpp-generator2/example/60_wrldtmpl/source/vis_09_file_helper.cpp
/home/martin/stage/cl-cpp-generator2/example/60_wrldtmpl/source/vis_10_math.cpp
/home/martin/stage/cl-cpp-generator2/example/60_wrldtmpl/source/vis_11_cl_buffer.cpp
/home/martin/stage/cl-cpp-generator2/example/60_wrldtmpl/source/vis_12_cl_kernel.cpp
/home/martin/stage/cl-cpp-generator2/example/60_wrldtmpl/source/vis_13_cl_helper.cpp
/home/martin/stage/cl-cpp-generator2/example/60_wrldtmpl/source/vis_14_scene_texture.cpp
/home/martin/stage/cl-cpp-generator2/example/60_wrldtmpl/source/vis_15_scene_material.cpp
/home/martin/stage/cl-cpp-generator2/example/60_wrldtmpl/source/vis_16_scene_matmanager.cpp
/home/martin/stage/cl-cpp-generator2/example/60_wrldtmpl/source/vis_17_scene_primitive.cpp
/home/martin/stage/cl-cpp-generator2/example/60_wrldtmpl/source/vis_18_scene.cpp )
add_executable( mytest ${SRCS} )
target_include_directories( mytest PUBLIC /home/martin/stage/cl-cpp-generator2/example/58_stdpar/source/ )
