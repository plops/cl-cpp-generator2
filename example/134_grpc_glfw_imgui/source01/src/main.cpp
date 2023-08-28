#include <GLFW/glfw3.h>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <glad/gl.h>
#include <glm/glm.hpp>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <iostream>
#include <linux/videodev2.h>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>
const char *vertexShaderSrc = R"(#version 450
layout (location=0) in vec2 aPos;

void main ()        {
            gl_Position=vec4(aPos, 1, 1);


}
 
)";

const char *fragmentShaderSrc = R"(#version 450
layout (location=0) out vec4 outColor;

void main ()        {
            outColor=vec4(1, 0, 0, 1);


}
 
)";

void message_callback(GLenum source, GLenum type, GLuint id, GLenum severity,
                      GLsizei length, GLchar const *message,
                      void const *user_param) {
  std::cout << "gl"
            << " source='" << source << "' "
            << " type='" << type << "' "
            << " id='" << id << "' "
            << " severity='" << severity << "' "
            << " message='" << message << "' " << std::endl;
}

int main(int argc, char **argv) {
  glfwInit();
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  auto window = glfwCreateWindow(800, 600, "v4l", nullptr, nullptr);
  if (!window) {
    throw std::runtime_error("Error creating glfw window");
  }

  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);
  if (!gladLoaderLoadGL()) {
    throw std::runtime_error("Error initializing glad");
  }
  std::cout << "get extensions" << std::endl;
  auto ext = glGetString(GL_EXTENSIONS);
  if (!(nullptr == ext)) {
    auto extstr = std::string(reinterpret_cast<const char *>(ext));
    std::cout << "extensions"
              << " extstr='" << extstr << "' " << std::endl;
  }

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init("#version 450 core");
  ImGui::StyleColorsClassic();

  glEnable(GL_CULL_FACE);
  glEnable(GL_DEBUG_OUTPUT);
  glDebugMessageCallback(message_callback, nullptr);

  std::cout << "Compile shader" << std::endl;
  auto success = 0;
  auto vertexShader = glCreateShader(GL_VERTEX_SHADER);
  glShaderSource(vertexShader, 1, &vertexShaderSrc, 0);
  glCompileShader(vertexShader);
  glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
  if (!success) {
    auto n = 512;
    auto infoLog = std::vector<char>(n);
    glGetShaderInfoLog(vertexShader, n, nullptr, infoLog.data());
    std::cout << "vertex shader compilation failed"
              << " std::string(infoLog.begin(), infoLog.end())='"
              << std::string(infoLog.begin(), infoLog.end()) << "' "
              << std::endl;

    exit(-1);
  }

  auto fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(fragmentShader, 1, &fragmentShaderSrc, 0);
  glCompileShader(fragmentShader);
  glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
  if (!success) {
    auto n = 512;
    auto infoLog = std::vector<char>(n);
    glGetShaderInfoLog(fragmentShader, n, nullptr, infoLog.data());
    std::cout << "fragment shader compilation failed"
              << " std::string(infoLog.begin(), infoLog.end())='"
              << std::string(infoLog.begin(), infoLog.end()) << "' "
              << std::endl;

    exit(-1);
  }

  auto program = glCreateProgram();
  glAttachShader(program, vertexShader);
  glAttachShader(program, fragmentShader);
  glLinkProgram(program);
  glGetProgramiv(program, GL_LINK_STATUS, &success);
  if (!success) {
    auto n = 512;
    auto infoLog = std::vector<char>(n);
    glGetShaderInfoLog(program, n, nullptr, infoLog.data());
    std::cout << "shader linking failed"
              << " std::string(infoLog.begin(), infoLog.end())='"
              << std::string(infoLog.begin(), infoLog.end()) << "' "
              << std::endl;

    exit(-1);
  }

  glDetachShader(program, vertexShader);
  glDetachShader(program, fragmentShader);

  glUseProgram(program);
  glClearColor(1, 1, 1, 1);

  auto texture = GLuint(0);
  glGenTextures(1, &texture);
  glBindTexture(GL_TEXTURE_2D, texture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    static bool showDemo = true;

    ImGui::ShowDemoWindow(&showDemo);
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    glfwSwapBuffers(window);
    glClear(GL_COLOR_BUFFER_BIT);
    std::this_thread::sleep_for(std::chrono::milliseconds(16));
  }
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
  glfwDestroyWindow(window);
  glfwTerminate();

  return 0;
}
