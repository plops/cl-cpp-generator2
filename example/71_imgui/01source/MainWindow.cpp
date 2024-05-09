// no preamble

#include <chrono>
#include <iomanip>
#include <iostream>
#include <thread>
extern std::chrono::time_point<std::chrono::high_resolution_clock> g_start_time;
#include "MainWindow.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_opengl3_loader.h"
#include "implot.h"
#include <GLFW/glfw3.h>
MainWindow::MainWindow()
    : show_demo_window_{true}, io{([&]() -> ImGuiIO & {
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImPlot::CreateContext();
        return ImGui::GetIO();
      })()} {
  //
}
MainWindow::~MainWindow() {
  //
}
void MainWindow::Init(GLFWwindow *window, const char *glsl_version) {
  //

  // enable keyboard controls, docking multi-viewport

  (io.ConfigFlags) = ((io.ConfigFlags) || (ImGuiConfigFlags_NavEnableKeyboard));
  // NavEnableKeyboard

  (io.ConfigFlags) = ((io.ConfigFlags) || (ImGuiConfigFlags_DockingEnable));
  // DockingEnable

  // setup ImGUI style

  ImGui::StyleColorsDark();
  auto style{ImGui::GetStyle()};
  if ((io.ConfigFlags) && (ImGuiConfigFlags_ViewportsEnable)) {
    (style.WindowRounding) = (0.F);
    (style.(Colors)[(ImGuiCol_WindowBg)].w) = (1.0F);
  }
  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init(glsl_version);
  auto font_fn{"nil"};
  auto font_size{16.F};
  auto *font{(io.Fonts)->(AddFontFromFileTTF(font_fn, font_size))};
  if ((font) == (nullptr)) {
    // loading font failed

  } else {
    // loaded font
  }
}
void MainWindow::NewFrame() {
  ImGui_ImplOpenGL3_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();
  ImGui::DockSpaceOverViewport();
}
void MainWindow::Update(std::function<void(void)> fun) {
  if (show_demo_window_) {
    ImGui::ShowDemoWindow(&show_demo_window_);
    ImPlot::ShowDemoWindow();
  }
  {
    ImGui::Begin("hello");
    ImGui::Checkbox("demo window", &show_demo_window_);
    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)",
                (1.00e+3F) / (ImGui::GetIO().Framerate),
                ImGui::GetIO().Framerate);
    ImGui::End();
  }
  fun();
  ImGui::EndFrame();
}
void MainWindow::Render(GLFWwindow *window) {
  auto screen_width{int(0)};
  auto screen_height{int(0)};
  glfwGetFramebufferSize(window, &screen_width, &screen_height);
  glViewport(0, 0, screen_width, screen_height);
  glClear(GL_COLOR_BUFFER_BIT);
  ImGui::Render();
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
  // update and render additional platform windows

  if ((io.ConfigFlags) && (ImGuiConfigFlags_ViewportsEnable)) {
    auto *backup_current_context{glfwGetCurrentContext()};
    ImGui::UpdatePlatformWindows();
    ImGui::RenderPlatformWindowsDefault();
    glfwMakeContextCurrent(backup_current_context);
  }
  glfwSwapBuffers(window);
}
void MainWindow::Shutdown() {
  //

  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
}