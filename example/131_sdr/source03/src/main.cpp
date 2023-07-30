#include "FFTWManager.h"
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
  auto maxStart = static_cast<int>(file.size() / sizeof(std::complex<short>));
  ImGui::SliderInt("Start", &start, 0, maxStart);

  static int windowSizeIndex = 4;
  auto itemsInt =
      std::vector<int>({32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384,
                        32768, 65536, 131072, 262144, 524288, 1048576});
  auto windowSize = itemsInt[windowSizeIndex];
  auto items = std::vector<std::string>(
      {"32", "64", "128", "256", "512", "1024", "2048", "4096", "8192", "16384",
       "32768", "65536", "131072", "262144", "524288", "1048576"});
  if (ImGui::BeginCombo("Window size", items[windowSizeIndex].c_str())) {
    for (auto i = 0; i < items.size(); i += 1) {
      auto is_selected = windowSizeIndex == i;
      if (ImGui::Selectable(items[i].c_str(), is_selected)) {
        windowSizeIndex = i;
        windowSize = itemsInt[windowSizeIndex];
      }
      if (is_selected) {
        ImGui::SetItemDefaultFocus();
      }
    }
    ImGui::EndCombo();
  }

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
    if (ImPlot::BeginPlot("Waveform (I/Q)")) {
      ImPlot::PlotLine("y1", x.data(), y1.data(), windowSize);
      ImPlot::PlotLine("y2", x.data(), y2.data(), windowSize);
      ImPlot::EndPlot();
    }

    static bool logScale = false;
    ImGui::Checkbox("Logarithmic Y-axis", &logScale);
    try {
      auto man = FFTWManager();
      auto in = std::vector<std::complex<double>>(windowSize);
      auto nyquist = windowSize / 2.0;
      auto sampleRate = 1.00e+7;
      for (auto i = 0; i < windowSize; i += 1) {
        x[i] = sampleRate *
               (static_cast<double>(i - (windowSize / 2)) / windowSize);
      }
      for (auto i = 0; i < windowSize; i += 1) {
        auto zs = file[(start + i)];
        auto zr = static_cast<double>(zs.real());
        auto zi = static_cast<double>(zs.imag());
        auto z = std::complex<double>(zr, zi);
        in[i] = z;
      }
      auto out = man.fft(in, windowSize);
      if (logScale) {
        for (auto i = 0; i < windowSize; i += 1) {
          y1[i] = 10. * log10(std::abs(out[i]));
        }
      } else {
        for (auto i = 0; i < windowSize; i += 1) {
          y1[i] = std::abs(out[i]);
        }
      }
      if (ImPlot::BeginPlot(logScale ? "FFT magnitude (dB)"
                                     : "FFT magnitude (linear)")) {
        ImPlot::PlotLine("y1", x.data(), y1.data(), windowSize);
        ImPlot::EndPlot();
      }

    } catch (const std::exception &e) {
      ImGui::Text("Error while processing FFT: %s", e.what());
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
