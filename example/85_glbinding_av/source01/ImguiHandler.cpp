// no preamble
#include <chrono>
#include <iomanip>
#include <iostream>
#include <thread>
extern std::chrono::time_point<std::chrono::high_resolution_clock> g_start_time;
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>
#include <imgui.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
void lprint(std::initializer_list<std::string> il, std::string file, int line,
            std::string fun);
#include "ImguiHandler.h"
ImguiHandler::ImguiHandler(GLFWwindow *window) {
  lprint({"initialize ImGui", " "}, __FILE__, __LINE__,
         &(__PRETTY_FUNCTION__[0]));
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  auto io = ImGui::GetIO();
  io.ConfigFlags = ((io.ConfigFlags) | (ImGuiConfigFlags_NavEnableKeyboard));
  ImGui::StyleColorsLight();
  {
    const auto installCallbacks = true;
    ImGui_ImplGlfw_InitForOpenGL(window, installCallbacks);
  }
  const auto glslVersion = "#version 150";
  ImGui_ImplOpenGL3_Init(glslVersion);
}
ImguiHandler::~ImguiHandler() {
  lprint({"Shutdown ImGui", " "}, __FILE__, __LINE__,
         &(__PRETTY_FUNCTION__[0]));
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
}