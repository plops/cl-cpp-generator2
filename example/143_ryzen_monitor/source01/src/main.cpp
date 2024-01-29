#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "implot.h"
#include <GLFW/glfw3.h>
#include <format>
#include <iostream>
extern "C" {
#include <libsmu.h>
#include <pm_tables.h>
#include <readinfo.h>
smu_obj_t obj;
};

void glfw_error_callback(int err, const char *description) {
  std::cout << std::format(" err='{}' description='{}'\n", err, description);
}

int main(int argc, char **argv) {
  glfwSetErrorCallback(glfw_error_callback);
  if (!glfwInit()) {
    std::cout << std::format("glfwInit failed\n");
    return 1;
  }
  auto glsl_version{"#version 130"};
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
  auto window{glfwCreateWindow(1280, 720, "ryzen_mon_glgui", nullptr, nullptr)};
  if (nullptr == window) {
    std::cout << std::format("Can't open glfw window\n");
    return 1;
  }
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImPlot::CreateContext();
  auto io{ImGui::GetIO()};
  io.ConfigFlags = io.ConfigFlags | ImGuiConfigFlags_NavEnableKeyboard;
  ImGui::StyleColorsDark();
  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init(glsl_version);
  auto show_demo_window{true};
  auto clear_color{ImVec4(0.40F, 0.50F, 0.60F, 1.0F)};
  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    if (show_demo_window) {
      ImGui::ShowDemoWindow(&show_demo_window);
      ImPlot::ShowDemoWindow();
    }
    ImGui::Render();
    auto w{0};
    auto h{0};
    glfwGetFramebufferSize(window, &w, &h);
    glViewport(0, 0, w, h);
    glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w,
                 clear_color.z * clear_color.w, clear_color.w);
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
