# Intermediate Graphics Library (IGL) Experiment

This repository is dedicated to experimenting with Facebook's
open-source Intermediate Graphics Library (IGL).

## About IGL

IGL is a real-time rendering library that supports a variety of
rendering backends:

- Metal 2+
- OpenGL 2.x (requires GL_ARB_framebuffer_object)
- OpenGL 3.1+
- OpenGL ES 2.0+
- Vulkan 1.1 (requires VK_KHR_buffer_device_address and VK_EXT_descriptor_indexing)
- WebGL 2.0

## Compilation

To compile the library, follow the steps below:

1. Clone the repository:
```bash
cd ~/src
git clone https://github.com/facebook/igl
```
Note: This operation downloads approximately 25MB of data.

2. Prepare the build:
```bash
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/home/martin/igl
```

This command also downloads additional dependencies: meshoptimizer,
glslang, glfw, stb, tinyobjloader, gli, glm, imgui, fmt. The size of
the directory after this operation is approximately 917MB.

## Exploring First Example: gen00

The first example that I explored was
[Tiny.cpp](https://github.com/facebook/igl/blob/main/samples/desktop/Tiny/Tiny.cpp).

- The example contains multiple parallel implementations and chooses
  between them using preprocessor ifdefs (vulkan, opengl, platforms).

- For this exploration, I decided to only implement the opengl
  version.

- Please note, the example requires a significant amount of code to
  simply draw a triangle. For those looking for a small graphics
  library that allows for drawing polygons with some shadows, this
  library may not be the best fit.
