#include <cstdlib>
#include <filesystem>
#include <glad/gl.h>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "glgui.grpc.pb.h"
#include <GLFW/glfw3.h>
#include <grpcpp/grpcpp.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
const char *const vertexShaderSrc = R"(#version 450
layout (location=0) in vec2 aPos;

void main ()        {
            gl_Position=vec4(aPos, 1, 1);


}
 
)";

const char *const fragmentShaderSrc = R"(#version 450
layout (location=0) out vec4 outColor;

void main ()        {
            outColor=vec4(1, 0, 0, 1);


}
 
)";

int main(int argc, char **argv) {
  grpc::ChannelArguments ch_args;
  // Increase max message size if needed

  ch_args.SetMaxReceiveMessageSize(-1);
  auto channel = grpc::CreateCustomChannel(
      "localhost:50051", grpc::InsecureChannelCredentials(), ch_args);
  auto stub = glgui::GLGuiService::NewStub(channel);

  auto request = glgui::RectangleRequest();
  auto response = glgui::RectangleResponse();

  auto context = grpc::ClientContext();

  auto status = stub->GetRandomRectangle(&context, request, &response);

  if (status.ok()) {

  } else {
  }

  glfwInit();
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  auto window = glfwCreateWindow(800, 600, "v4l", nullptr, nullptr);
  if (!window) {

    return -1;
  }

  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);
  if (!gladLoaderLoadGL()) {

    return -2;
  }

  if (auto ext = glGetString(GL_EXTENSIONS); nullptr != ext) {
    auto extstr = std::string(reinterpret_cast<const char *>(ext));
  }

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init("#version 450 core");
  ImGui::StyleColorsClassic();

  glEnable(GL_CULL_FACE);

  auto success = 0;
  auto vertexShader = glCreateShader(GL_VERTEX_SHADER);
  glShaderSource(vertexShader, 1, &vertexShaderSrc, nullptr);
  glCompileShader(vertexShader);
  glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
  if (!success) {
    exit(-1);
  }

  auto fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(fragmentShader, 1, &fragmentShaderSrc, nullptr);
  glCompileShader(fragmentShader);
  glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
  if (!success) {
    exit(-1);
  }

  auto program = glCreateProgram();
  glAttachShader(program, vertexShader);
  glAttachShader(program, fragmentShader);
  glLinkProgram(program);
  glGetProgramiv(program, GL_LINK_STATUS, &success);
  if (!success) {
    exit(-1);
  }

  glDetachShader(program, vertexShader);
  glDetachShader(program, fragmentShader);

  glUseProgram(program);
  glClearColor(1, 1, 1, 1);

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
