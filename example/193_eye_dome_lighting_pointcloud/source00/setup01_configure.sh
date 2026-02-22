mkdir b
cd b
cmake .. \
      -GNinja \
      -DCMAKE_BUILD_TYPE=Release \
      -DGLFW_BUILD_WAYLAND=OFF \
      -DGLFW_INSTALL=OFF \
      -DOpenGL_GL_PREFERENCE=GLVND
