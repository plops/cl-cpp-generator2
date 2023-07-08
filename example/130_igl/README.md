# Trying out IGL by facebook

- it is a library to render realtime
- open-source Intermediate Graphics Library (IGL)

- supports the following rendering backends

Metal 2+
OpenGL 2.x (requires GL_ARB_framebuffer_object)
OpenGL 3.1+
OpenGL ES 2.0+
Vulkan 1.1 (requires VK_KHR_buffer_device_address and VK_EXT_descriptor_indexing)
WebGL 2.0


## compile the library

```
cd ~/src
git clone https://github.com/facebook/igl
# this downloads 25MB 

cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/home/martin/igl

```
- cmake this downloads some more dependencies: meshoptimizer, glslang,
 glfw, stb, tinyobjloader, gli, glm, imgui, fmt ...

- after this the directory is 917MB


## first example gen00

- go through
  https://github.com/facebook/igl/blob/main/samples/desktop/Tiny/Tiny.cpp

-  it has many parallel implementations and chooses between them with
  preprocessor ifdefs (vulkan, opengl, platforms)

- i decided to only implement opengl

- it requires a rediculus amount of code to draw a triangle. ideally i
  want a small graphics library that allows me to draw polygons with
  some shadows. i think this is not it.
