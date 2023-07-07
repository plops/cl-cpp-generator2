cmake_minimum_required(VERSION 3.16)
project(MainProject)

add_executable(main main.cpp)

target_include_directories(main PRIVATE /home/martin/src/igl/src)

target_link_libraries(main PRIVATE 
    /home/martin/src/igl/b/shell/windows/libIGLShellPlatform.a
    /home/martin/src/igl/b/shell/windows/libIGLShellApp_opengl.a
    /home/martin/src/igl/b/shell/libIGLShellShared.a
    /home/martin/src/igl/b/src/igl/opengl/libIGLOpenGL.a
    /home/martin/src/igl/b/src/igl/libIGLLibrary.a
    /home/martin/src/igl/b/IGLU/libIGLUmanagedUniformBuffer.a
    /home/martin/src/igl/b/IGLU/libIGLUimgui.a
    /home/martin/src/igl/b/IGLU/libIGLUuniform.a
    /home/martin/src/igl/b/IGLU/libIGLUsimple_renderer.a
    /home/martin/src/igl/b/IGLU/libIGLUtexture_accessor.a
    /home/martin/src/igl/b/third-party/deps/src/bc7enc/libbc7enc.a
    /home/martin/src/igl/b/third-party/deps/src/glfw/src/libglfw3.a
    /home/martin/src/igl/b/libIGLstb.a
    /home/martin/igl/lib/libglfw3.a
    /home/martin/igl/lib/libfmt.a
    /home/martin/igl/lib/libtinyobjloader.a
    /home/martin/igl/lib/libmeshoptimizer.a
    X11 
    GL 
    EGL
)

target_compile_definitions(main PRIVATE IGL_CMAKE_BUILD)
