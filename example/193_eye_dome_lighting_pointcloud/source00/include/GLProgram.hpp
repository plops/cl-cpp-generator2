#pragma once
#include <glad/glad.h>
#include <glm/glm.hpp>

// RAII Wrapper for OpenGL Shader Programs
class GLProgram {
public:
    GLuint id{0};

    GLProgram(const char* vsSource, const char* fsSource);
    ~GLProgram();

    // Delete copy semantics to prevent double-freeing OpenGL resources
    GLProgram(const GLProgram&) = delete;
    GLProgram& operator=(const GLProgram&) = delete;

    void use() const;

    void setUniform(const char* name, float value) const;
    void setUniform(const char* name, int value) const;
    void setUniform(const char* name, const glm::vec2& value) const;
    void setUniform(const char* name, const glm::vec3& value) const;
    void setUniform(const char* name, const glm::mat4& value) const;

private:
    [[nodiscard]] static GLuint compileShader(GLenum type, const char* source);
};
