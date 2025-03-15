// no preamble
#include <chrono>
#include <iostream>
#include <spdlog/spdlog.h>
#include <thread>
extern const std::chrono::time_point<std::chrono::high_resolution_clock>
    g_start_time;
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>
#include <imgui.h>
#define GLFW_INCLUDE_NONE
#include "ImguiHandler.h"
#include <GLFW/glfw3.h>
ImguiHandler::ImguiHandler(GLFWwindow *window) {
  spdlog::info("initialize ImGui");
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGui::StyleColorsLight();
  {
    const auto installCallbacks{true};
    ImGui_ImplGlfw_InitForOpenGL(window, installCallbacks);
  }
  const auto glslVersion{"#version 150"};
  ImGui_ImplOpenGL3_Init(glslVersion);
  auto &io{ImGui::GetIO()};
  spdlog::info("enable keyboard input for imgui");
  (io.ConfigFlags) = ((io.ConfigFlags) || (ImGuiConfigFlags_NavEnableKeyboard));
}
void ImguiHandler::NewFrame() {
  ImGui_ImplOpenGL3_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();
  auto showDemoWindow{true};
  ImGui::ShowDemoWindow(&showDemoWindow);
}
void ImguiHandler::Render() { ImGui::Render(); }
void ImguiHandler::RenderDrawData() {
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}
void ImguiHandler::Begin(const char *str) { ImGui::Begin(str); }
void ImguiHandler::End() { ImGui::End(); }
void ImguiHandler::Image(uint tex, int w, int h) {
  ImGui::Image(reinterpret_cast<void *>(static_cast<intptr_t>(tex)),
               ImVec2(static_cast<float>(w), static_cast<float>(h)));
}
void ImguiHandler::SliderFloat(const char *label, float *val, float min,
                               float max, const char *fmt) {
  ImGui::SliderFloat(label, val, min, max, fmt);
}
ImguiHandler::~ImguiHandler() {
  spdlog::info("Shutdown ImGui");
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
}