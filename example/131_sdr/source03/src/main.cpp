#include "GLFW/glfw3.h"
#include "GpsCACodeGenerator.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "implot.h"
#include <cmath>
#include <iostream>
#include <vector>

void glfw_error_callback(int err, const char *desc) {
  std::cout << "GLFW erro:"
            << " err='" << err << "' "
            << " desc='" << desc << "' " << std::endl;
}

void DemoImplot() {
  static std::vector<double> x;
  static std::vector<double> y1;
  static std::vector<double> y2;
  if (x.empty()) {
    for (auto i = 0; i < 1000; i += 1) {
      auto x_ = 3.14159265358979300000000000000 * (4.0 / 1.00e+3) * i;
      x.push_back(x_);
      y1.push_back(cos(x_));
      y2.push_back(sin(x_));
    }
  }
  if (ImPlot::BeginPlot("Plot")) {
    ImPlot::PlotLine("y1", x.data(), y1.data(), static_cast<int>(x.size()));
    ImPlot::PlotLine("y2", x.data(), y2.data(), static_cast<int>(x.size()));
    ImPlot::EndPlot();
  }
}

int main(int argc, char **argv) {
  glfwSetErrorCallback(glfw_error_callback);
  if (0 == glfwInit()) {
    return 1;
  }
  const auto *glsl_version = "#version 130";
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

  auto *window = glfwCreateWindow(800, 600, "imgui_dsp", nullptr, nullptr);
  if (nullptr == window) {
    return 1;
  }
  glfwMakeContextCurrent(window);
  std::cout << "enable vsync" << std::endl;
  glfwSwapInterval(1);
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImPlot::CreateContext();
  auto &io = ImGui::GetIO();
  io.ConfigFlags = io.ConfigFlags | ImGuiConfigFlags_NavEnableKeyboard;

  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init(glsl_version);

  auto ca = GpsCACodeGenerator(4);
  std::cout << "CA" << std::endl;
  GpsCACodeGenerator::print_square(ca.generate_sequence(1023));

  while (glfwWindowShouldClose(window) == 0) {
    glfwPollEvents();
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    DemoImplot();
    ImGui::Render();
    auto w = 0;
    auto h = 0;
    glfwGetFramebufferSize(window, &w, &h);
    glViewport(0, 0, w, h);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    glfwSwapBuffers(window);
  }
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImPlot::DestroyContext();
  ImGui::DestroyContext();
  glfwDestroyWindow(window);
  glfwTerminate();

  return 0;
}
