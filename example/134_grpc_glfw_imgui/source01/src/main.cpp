#include <cstdlib>
#include <filesystem>
#include <future>
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

void message_callback(GLenum source, GLenum type, GLuint id, GLenum severity,
                      [[maybe_unused]] GLsizei length, GLchar const *message,
                      [[maybe_unused]] void const *user_param) {
  std::cout << "gl source='" << source << "' type='" << type << "' id='" << id
            << "' severity='" << severity << "' message='" << message << "'\n";
}

int main(int argc, char **argv) {
  grpc::ChannelArguments ch_args;
  // Increase max message size if needed

  ch_args.SetMaxReceiveMessageSize(-1);
  auto channel = grpc::CreateCustomChannel(
      "localhost:50051", grpc::InsecureChannelCredentials(), ch_args);
  auto stub = glgui::GLGuiService::NewStub(channel);
  auto get_random_rectangle =
      [](std::unique_ptr<glgui::GLGuiService::Stub> const &stub_) {
        auto request = glgui::RectangleRequest();
        auto response = glgui::RectangleResponse();
        auto context = grpc::ClientContext();
        auto status = stub_->GetRandomRectangle(&context, request, &response);
        if (status.ok()) {
          std::cout << " response.x1()='" << response.x1() << "'\n";
        } else {
          std::cout << " status.error_message()='" << status.error_message()
                    << "'\n";
        }
      };
  get_random_rectangle(stub);
  auto get_image = [](std::unique_ptr<glgui::GLGuiService::Stub> const &stub_) {
    auto request = glgui::GetImageRequest();
    auto response = glgui::GetImageResponse();
    request.set_width(128);
    request.set_height(128);
    auto context = grpc::ClientContext();
    auto status = stub_->GetImage(&context, request, &response);
    if (status.ok()) {
      return response;
    } else {
      throw std::runtime_error(status.error_message());
    }
  };
  glfwInit();
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  auto window = glfwCreateWindow(800, 600, "v4l", nullptr, nullptr);
  if (!window) {
    std::cout << "Error creating glfw window.\n";
    return -1;
  }
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);
  if (!gladLoaderLoadGL()) {
    std::cout << "Error initializing glad.\n";
    return -2;
  }
  std::cout << "Get extensions.\n";
  if (auto ext = glGetString(GL_EXTENSIONS); nullptr != ext) {
    auto extension_str = std::string(reinterpret_cast<const char *>(ext));
    std::cout << "extension extension_str='" << extension_str << "'\n";
  }
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init("#version 450 core");
  ImGui::StyleColorsClassic();
  glEnable(GL_CULL_FACE);
  glEnable(GL_DEBUG_OUTPUT);
  glDebugMessageCallback(message_callback, nullptr);
  std::cout << "Compile shader.\n";
  auto success = 0;
  auto vertexShader = glCreateShader(GL_VERTEX_SHADER);
  glShaderSource(vertexShader, 1, &vertexShaderSrc, nullptr);
  glCompileShader(vertexShader);
  glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
  if (!success) {
    auto infoLog = std::vector<char>(512);
    glGetShaderInfoLog(vertexShader, static_cast<GLsizei>(infoLog.size()),
                       nullptr, infoLog.data());
    auto info = std::string(infoLog.begin(), infoLog.end());
    std::cout << "Vertex shader compilation failed. info='" << info << "'\n";
    exit(-1);
  }
  auto fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(fragmentShader, 1, &fragmentShaderSrc, nullptr);
  glCompileShader(fragmentShader);
  glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
  if (!success) {
    auto infoLog = std::vector<char>(512);
    glGetShaderInfoLog(fragmentShader, static_cast<GLsizei>(infoLog.size()),
                       nullptr, infoLog.data());
    auto info = std::string(infoLog.begin(), infoLog.end());
    std::cout << "Fragment shader compilation failed. info='" << info << "'\n";
    exit(-1);
  }
  auto program = glCreateProgram();
  glAttachShader(program, vertexShader);
  glAttachShader(program, fragmentShader);
  glLinkProgram(program);
  glGetProgramiv(program, GL_LINK_STATUS, &success);
  if (!success) {
    auto infoLog = std::vector<char>(512);
    glGetShaderInfoLog(program, static_cast<GLsizei>(infoLog.size()), nullptr,
                       infoLog.data());
    auto info = std::string(infoLog.begin(), infoLog.end());
    std::cout << "Shader linking failed. info='" << info << "'\n";
    exit(-1);
  }
  glDetachShader(program, vertexShader);
  glDetachShader(program, fragmentShader);
  glUseProgram(program);
  glClearColor(1, 1, 1, 1);
  auto texture = GLuint(0);
  auto texture_w = 0;
  auto texture_h = 0;
  glGenTextures(1, &texture);
  glBindTexture(GL_TEXTURE_2D, texture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  auto future = std::future<glgui::GetImageResponse>();
  auto update_texture_if_ready =
      [&texture_w, &texture_h, texture,
       get_image](std::unique_ptr<glgui::GLGuiService::Stub> &stub_,
                  std::future<glgui::GetImageResponse> &future_) {
        if (future_.valid()) {
          if (future_.wait_for(std::chrono::seconds(0)) ==
              std::future_status::ready) {
            auto response = future_.get();
            glBindTexture(GL_TEXTURE_2D, texture);
            texture_w = response.width();
            texture_h = response.height();
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, texture_w, texture_h, 0,
                         GL_RGB, GL_UNSIGNED_BYTE, response.data().c_str());
            future_ = std::future<glgui::GetImageResponse>();
          }
        } else {
          future_ = std::async(std::launch::async, get_image, std::ref(stub_));
        }
      };
  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    update_texture_if_ready(stub, future);
    glBindTexture(GL_TEXTURE_2D, texture);
    ImGui::Begin("texture");
    ImGui::Image(
        reinterpret_cast<void *>(static_cast<intptr_t>(texture)),
        ImVec2(static_cast<float>(texture_w), static_cast<float>(texture_h)));
    ImGui::End();
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
