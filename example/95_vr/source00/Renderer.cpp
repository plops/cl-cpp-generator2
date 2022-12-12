// no preamble
#include "Renderer.h"
#include "bah.h"
Renderer::Renderer(GLsizei width, GLsizei height)
    : program(Program()), geometry(Geometry()) {
  for (auto i = 0; (i) < (VRAPI_FRAME_LAYER_EYE_MAX); (i) += (1)) {
    framebuffers.push_back(Framebuffer(width, height));
  }
}