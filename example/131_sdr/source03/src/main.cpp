#include "GLFW/glfw3.h"
#include "GpsCACodeGenerator.h"
#include "MemoryMappedComplexShortFile.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "implot.h"
#include <cmath>
#include <iostream>
#include <vector>

void glfw_error_callback(int err, const char *desc) {}

void DrawPlot(const MemoryMappedComplexShortFile &file) {
  static int start = 0;
  static int windowSize = 512;
  auto maxStart = static_cast<int>(file.size() / sizeof(std::complex<short>));
  ImGui::SliderInt("Start", &start, 0, maxStart);
  ImGui::InputInt("Window size", &windowSize);

  if (start + windowSize <= maxStart && 0 < windowSize) {
    auto x = std::vector<double>(windowSize);
    auto y1 = std::vector<double>(windowSize);
    auto y2 = std::vector<double>(windowSize);
    for (auto i = 0; i < windowSize; i += 1) {
      auto z = file[(start + i)];
      x[i] = i;
      y1[i] = z.real();
      y2[i] = z.imag();
    }
    if (ImPlot::BeginPlot("Plot")) {
      ImPlot::PlotLine("y1", x.data(), y1.data(), windowSize);
      ImPlot::PlotLine("y2", x.data(), y2.data(), windowSize);
      ImPlot::EndPlot();
    }
  }
}

int main(int argc, char **argv) {
  glfwSetErrorCallback(glfw_error_callback);
  if (0 == glfwInit()) {

    return 1;
  }
  auto *window = glfwCreateWindow(800, 600, "imgui_dsp", nullptr, nullptr);
  if (nullptr == window) {

    return 1;
  }
  glfwMakeContextCurrent(window);

  glfwSwapInterval(1);
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImPlot::CreateContext();
  auto &io = ImGui::GetIO();
  io.ConfigFlags = io.ConfigFlags | ImGuiConfigFlags_NavEnableKeyboard;

  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init("#version 130");
  glClearColor(0, 0, 0, 1);

  try {
    auto fn = "/mnt5/capturedData_L1_rate10MHz_bw5MHz_iq_short.bin";
    auto file = MemoryMappedComplexShortFile(fn);

    while (!glfwWindowShouldClose(window)) {
      glfwPollEvents();
      ImGui_ImplOpenGL3_NewFrame();
      ImGui_ImplGlfw_NewFrame();
      ImGui::NewFrame();
      DrawPlot(file);
      ImGui::Render();
      auto w = 0;
      auto h = 0;
      glfwGetFramebufferSize(window, &w, &h);
      glViewport(0, 0, w, h);
      glClear(GL_COLOR_BUFFER_BIT);

      ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
      glfwSwapBuffers(window);
    }

  } catch (const std::runtime_error &e) {

    return -1;
  }
  return 0;
}
