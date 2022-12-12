// no preamble
#include "Geometry.h"
#include "bah.h"
Geometry::Geometry() {
  glGenVertexArrays(1, vertex_array);
  glBindVertexArrays(vertex_array);
  glGenBuffers(1, vertex_buffer);
  glBufferData(GL_ARRAY_BUFFER, cube.vertices.size, cube.vertices.data(),
               GL_STATIC_DRAW);
}