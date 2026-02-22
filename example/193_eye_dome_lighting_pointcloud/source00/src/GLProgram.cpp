#include "GLProgram.hpp"
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <string>

GLProgram::GLProgram(const char* vsSource, const char* fsSource) {
    auto vs = compileShader(GL_VERTEX_SHADER, vsSource);
    auto fs = compileShader(GL_FRAGMENT_SHADER, fsSource);

    id = glCreateProgram();
    glAttachShader(id, vs);
    glAttachShader(id, fs);
    glLinkProgram(id);

    glDeleteShader(vs);
    glDeleteShader(fs);
}

GLProgram::~GLProgram() {
    if (id != 0) glDeleteProgram(id);
}

void GLProgram::use() const {
    glUseProgram(id);
}

void GLProgram::setUniform(const char* name, float value) const {
    glUniform1f(glGetUniformLocation(id, name), value);
}

void GLProgram::setUniform(const char* name, int value) const {
    glUniform1i(glGetUniformLocation(id, name), value);
}

void GLProgram::setUniform(const char* name, const glm::vec2& value) const {
    glUniform2fv(glGetUniformLocation(id, name), 1, glm::value_ptr(value));
}

void GLProgram::setUniform(const char* name, const glm::vec3& value) const {
    glUniform3fv(glGetUniformLocation(id, name), 1, glm::value_ptr(value));
}

void GLProgram::setUniform(const char* name, const glm::mat4& value) const {
    glUniformMatrix4fv(glGetUniformLocation(id, name), 1, GL_FALSE, glm::value_ptr(value));
}

GLuint GLProgram::compileShader(GLenum type, const char* source) {
    auto shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    auto success = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        auto infoLog = std::string(512, '\0');
        glGetShaderInfoLog(shader, 512, nullptr, infoLog.data());
        std::cerr << "Shader Compilation Failed:\n" << infoLog << std::endl;
    }
    return shader;
}
