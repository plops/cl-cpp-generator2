
#include "utils.h"

#include "globals.h"

extern State state;
// implementation

class mat4;
Shader::Shader(const char *vfile, const char *pfile, bool fromString) {
  nil;
  if (fromString) {
    Compile(vfile, pfile);
  } else {
    Init(vfile, pfile);
  }
  nil;
}
Shader::~Shader() {
  nil;
  glDetachShader(ID, pixel);
  glDetachShader(ID, vertex);
  glDeleteShader(pixel);
  glDeleteShader(vertex);
  glDeleteProgram(ID);
  CheckGL();
}
void Shader::Init(const char *vfile, const char *pfile) {
  nil;
  auto vsText{TextFileRead(vfile)};
  auto fsText{TextFileRead(pfile)};
  if ((0) == (vsText.size())) {
    {

      auto lock{std::unique_lock<std::mutex>(state._stdout_mutex)};
      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ") << ("File not found") << (" ") << (std::setw(8))
                  << (" vfile='") << (vfile) << ("'") << (std::endl)
                  << (std::flush);
    }
  }
  if ((0) == (fsText.size())) {
    {

      auto lock{std::unique_lock<std::mutex>(state._stdout_mutex)};
      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ") << ("File not found") << (" ") << (std::setw(8))
                  << (" pfile='") << (pfile) << ("'") << (std::endl)
                  << (std::flush);
    }
  }
  auto vertexText{vsText.c_str()};
  auto fragmentText{fsText.c_str()};
  Compile(vertexText, fragmentText);
  nil;
}
void Shader::Compile(const char *vtext, const char *ftext) {
  nil;
  auto vertex{glCreateShader(GL_VERTEX_SHADER)};
  auto pixel{glCreateShader(GL_FRAGMENT_SHADER)};
  glShaderSource(vertex, 1, &vtext, 0);
  glCompileShader(vertex);
  CheckShader(vertex, vtext, ftext);
  glShaderSource(pixel, 1, &ftext, 0);
  glCompileShader(pixel);
  CheckShader(pixel, vtext, ftext);
  (ID) = (glCreateProgram());
  glAttachShader(ID, vertex);
  glAttachShader(ID, pixel);
  glBindAttribLocation(ID, 0, "pos");
  glBindAttribLocation(ID, 1, "tuv");
  glLinkProgram(ID);
  glCheckProgram(ID, vtext, ftext);
  CheckGL();
}
void Shader::Bind() {
  nil;
  glUseProgram(ID);
  CheckGL();
}
void Shader::Unbind() {
  nil;
  glUseProgram(0);
  CheckGL();
}
void Shader::SetInputTexture(uint slot, const char *name, GLTexture *texture) {
  nil;
  glActiveTexture((GL_TEXTURE0) + (slot));
  glBindTexture(GL_TEXTURE_2D, texture->ID);
  glUniform1i(glGetUniformLocation(ID, name), slot);
  CheckGL();
  CheckGL();
}
void Shader::SetInputMatrix(const char *name, const mat4 &matrix) {
  nil;
  auto data{static_cast<const GLfloat *>(&matrix)};
  glUniformMatrix4fv(glGetUniformLocation(ID, name), 1, GL_FALSE, data);
  CheckGL();
}
void Shader::SetFloat(const char *name, const float v) {
  nil;
  glUniform1f(glGetUniformLocation(ID, name), v);
  CheckGL();
}
void Shader::SetInt(const char *name, const int v) {
  nil;
  glUniform1i(glGetUniformLocation(ID, name), v);
  CheckGL();
}
void Shader::SetUInt(const char *name, const uint v) {
  nil;
  glUniform1ui(glGetUniformLocation(ID, name), v);
  CheckGL();
}