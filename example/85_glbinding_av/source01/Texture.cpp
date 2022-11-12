// no preamble
#include <chrono>
#include <iostream>
#include <thread>
void lprint(std::initializer_list<std::string> il, std::string file, int line,
            std::string fun);
extern const std::chrono::time_point<std::chrono::high_resolution_clock>
    g_start_time;
#include <glbinding/gl32core/gl.h>
#include <glbinding/glbinding.h>
using namespace gl32core;
using namespace glbinding;
#include "Texture.h"
unsigned int Texture::GetImageTexture() { return image_texture; }
int Texture::GetWidth() { return m_width; }
int Texture::GetHeight() { return m_height; }
Texture::Texture(int w, int h, int internalFormat)
    : image_texture(0), initialized_p(false), m_internalFormat(internalFormat),
      m_width(w), m_height(h) {
  Reset(nullptr, w, h, internalFormat);
}
bool Texture::Compatible_p(int w, int h, unsigned int internalFormat) {
  return (((m_internalFormat) == (internalFormat)) && ((w) == (m_width)) &&
          ((h) == (m_height)));
}
void Texture::Reset(unsigned char *data, int w, int h, int internalFormat) {
  if (((initialized_p) && (!(Compatible_p(w, h, internalFormat))))) {
    glDeleteTextures(1, &image_texture);
    initialized_p = false;
    glGenTextures(1, &image_texture);
  }
  if (!(initialized_p)) {
    glBindTexture(GL_TEXTURE_2D, image_texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, w, h, 0,
                 GLenum::GL_LUMINANCE, GL_UNSIGNED_BYTE, nullptr);
    m_internalWidth = w;
    m_internalHeight = h;
  }
  glBindTexture(GL_TEXTURE_2D, image_texture);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GLenum::GL_LUMINANCE,
                  GL_UNSIGNED_BYTE, data);
  m_width = w;
  m_height = h;
  initialized_p = true;
}
Texture::~Texture() {
  if (initialized_p) {
    initialized_p = false;
    glDeleteTextures(1, &image_texture);
  }
}