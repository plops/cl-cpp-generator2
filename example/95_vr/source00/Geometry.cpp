// no preamble
#include "Geometry.h"
#include "App.h"
#include "AttribPointer.h"
#include "Cube.h"
#include "Egl.h"
#include "Framebuffer.h"
#include "Program.h"
#include "Renderer.h"
#include "Vertex.h"
#include "VrApi.h"
#include "VrApi_Helpers.h"
#include "VrApi_Input.h"
#include "VrApi_SystemUtils.h"
#include "android_native_app_glue.h"
#include "bah.h"
#include "core.h"
#include "format-inl.h"
#include "format.h"
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>
#include <android/log.h>
#include <android/window.h>
#include <cstdin>
#include <cstdlib>
#include <iostream>
#include <unistd.h>
extern static const std::array<AttribPointer, 2> ATTRIB_POINTERS;
#include "Geometry.h"
Geometry::Geometry() {
  glGenVertexArrays(1, &vertex_array);
  glBindVertexArrays(vertex_array);
  glGenBuffers(1, &vertex_buffer);
  glBufferData(GL_ARRAY_BUFFER, cube.vertices.size, cube.vertices.data(),
               GL_STATIC_DRAW);
  auto i = 0;
  for (auto attrib : ATTRIB_POINTERS) {
    glEnableVertexAttribArray(i);
    glVertexAttribPointer(i, attrib.size, attrib.type, attrib.normalized,
                          attrib.stride, attrib.pointer);
    (i)++;
  }
  glGenBuffers(1, &index_buffer);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, cube.indices.size(), cube.indices,
               GL_STATIC_DRAW);
  glBindVertexArray(0);
}
Geometry::~Geometry() {
  glDeleteBuffers(1, &index_buffer);
  glDeleteBuffers(1, &vertex_buffer);
  glDeleteVertexArrays(1, &vertex_array);
}