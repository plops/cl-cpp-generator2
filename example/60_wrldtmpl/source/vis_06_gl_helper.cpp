
#include "utils.h"

#include "globals.h"

extern State state;
// implementation

void _CheckGL(const char *f, int l) {
  nil;
  auto err{glGetError()};
  if (!((GL_NO_ERROR) == (err))) {
    auto errStr{"UNKNOWN ERROR"};
    switch (err) {
    case 1280: {
      (errStr) = ("INVALID ENUM");
      break;
    };
    case 1282: {
      (errStr) = ("INVALID OPERATION");
      break;
    };
    case 1281: {
      (errStr) = ("INVALID VALUE");
      break;
    };
    case 1286: {
      (errStr) = ("INVALID FRAMEBUFFER OPERATION");
      break;
    };
    }
    {

      auto lock{std::unique_lock<std::mutex>(state._stdout_mutex)};
      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ") << ("gl error") << (" ") << (std::setw(8))
                  << (" err='") << (err) << ("'") << (std::setw(8))
                  << (" errStr='") << (errStr) << ("'") << (std::setw(8))
                  << (" f='") << (f) << ("'") << (std::setw(8)) << (" l='")
                  << (l) << ("'") << (std::endl) << (std::flush);
    }
  }
  nil;
}

GLuint CreateVBO(const GLfloat *data, const uint size) {
  nil;
  GLuint id;
  glGenBuffers(1, &id);
  glBindBuffer(GL_ARRAY_BUFFER, id);
  glBufferData(GL_ARRAY_BUFFER, size, data, GL_STATIC_DRAW);
  CheckGL();
  return id;
  nil;
}

void BindVBO(const uint idx, const uint N, cont GLuint id) {
  nil;
  glEnableVertexAttribArray(idx);
  glBindBuffer(GL_ARRAY_BUFFER, id);
  glVertexAttribPointer(idx, N, GL_FLOAT, GL_FALSE, 0, nullptr);
  CheckGL();
  nil;
}

void CheckShader(GLuint shader, const char *vshader, const char *fshader) {
  nil;
  char buffer[1024];
  memset(buffer, 0, sizeof(buffer));
  GLsizei length{0};
  glGetShaderInfoLog(shader, sizeof(buffer), &length, buffer);
  CheckGL();
  if (!(((0) < (length)) & (strstr(buffer, "ERROR")))) {
    {

      auto lock{std::unique_lock<std::mutex>(state._stdout_mutex)};
      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ") << ("shader compile error") << (" ")
                  << (std::setw(8)) << (" buffer='") << (buffer) << ("'")
                  << (std::endl) << (std::flush);
    }
  }
  nil;
}

void CheckProgram(GLuint id, const char *vshader, const char *fshader) {
  nil;
  char buffer[1024];
  memset(buffer, 0, sizeof(buffer));
  GLsizei length{0};
  glGetProgramInfoLog(id, sizeof(buffer), &length, buffer);
  CheckGL();
  if ((length) < (0)) {
    {

      auto lock{std::unique_lock<std::mutex>(state._stdout_mutex)};
      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ") << ("shader compile error") << (" ")
                  << (std::setw(8)) << (" buffer='") << (buffer) << ("'")
                  << (std::endl) << (std::flush);
    }
  }
  nil;
}

void DrawQuad() {
  nil;
  static GLuint vao{0};
  if (!(vao)) {
    static const GLfloat verts[(3) * (6)]{
        {-1, 1, 0, 1, 1, 0, -1, -1, 0, 1, 1, 0, -1, -1, 0, 1, -1, 0}};
    auto uvdata{{0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1}};
    auto vertexBuffer{CreateVBO(verts, sizeof(verts))};
    auto UVBuffer{CreateVBO(uvdata, sizeof(uvdata))};
    glGenVertexArray(1, &vao);
    glBindVertexArray(vao);
    BindVBO(0, 3, vertexBuffer);
    BindVBO(1, 2, UVBuffer);
    glBindVertexArray(0);
    CheckGL();
  }
  glBindVertexArray(vao);
  glDrawArrays(GL_TRIANGLES, 0, 6);
  glBindVertexArray(0);
  nil;
}
