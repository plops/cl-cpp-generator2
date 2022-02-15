// no preamble
;
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
    : show_demo_window_(true), io(([]() -> ImGuiIO & {
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImPlot::CreateContext();
        return ImGui::GetIO();
      })()) {
  {

    std::chrono::duration<double> timestamp =
        std::chrono::high_resolution_clock::now() - g_start_time;
    (std::cout) << (std::setw(10)) << (timestamp.count()) << (" ")
                << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
                << (__LINE__) << (" ") << (__func__) << (" ") << ("") << (" ")
                << (std::endl) << (std::flush);
  }
}
MainWindow::~MainWindow() {
  {

    std::chrono::duration<double> timestamp =
        std::chrono::high_resolution_clock::now() - g_start_time;
    (std::cout) << (std::setw(10)) << (timestamp.count()) << (" ")
                << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
                << (__LINE__) << (" ") << (__func__) << (" ") << ("") << (" ")
                << (std::endl) << (std::flush);
  }
}
void MainWindow::Init(std::shared_ptr<GLFWwindow> window,
                      const char *glsl_version) {
  {

    std::chrono::duration<double> timestamp =
        std::chrono::high_resolution_clock::now() - g_start_time;
    (std::cout) << (std::setw(10)) << (timestamp.count()) << (" ")
                << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
                << (__LINE__) << (" ") << (__func__) << (" ") << ("") << (" ")
                << (std::endl) << (std::flush);
  }
  // enable keyboard controls, docking multi-viewport
  ;
  io.ConfigFlags = ((io.ConfigFlags) | (ImGuiConfigFlags_NavEnableKeyboard));
  {

    std::chrono::duration<double> timestamp =
        std::chrono::high_resolution_clock::now() - g_start_time;
    (std::cout) << (std::setw(10)) << (timestamp.count()) << (" ")
                << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
                << (__LINE__) << (" ") << (__func__) << (" ")
                << ("NavEnableKeyboard") << (" ") << (std::setw(8))
                << (" ImGuiConfigFlags_NavEnableKeyboard='")
                << (ImGuiConfigFlags_NavEnableKeyboard) << ("'")
                << (std::setw(8)) << (" io.ConfigFlags='") << (io.ConfigFlags)
                << ("'") << (std::setw(8)) << (" ImGui::GetIO().ConfigFlags='")
                << (ImGui::GetIO().ConfigFlags) << ("'") << (std::endl)
                << (std::flush);
  }
  io.ConfigFlags = ((io.ConfigFlags) | (ImGuiConfigFlags_DockingEnable));
  {

    std::chrono::duration<double> timestamp =
        std::chrono::high_resolution_clock::now() - g_start_time;
    (std::cout) << (std::setw(10)) << (timestamp.count()) << (" ")
                << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
                << (__LINE__) << (" ") << (__func__) << (" ")
                << ("DockingEnable") << (" ") << (std::setw(8))
                << (" ImGuiConfigFlags_DockingEnable='")
                << (ImGuiConfigFlags_DockingEnable) << ("'") << (std::setw(8))
                << (" io.ConfigFlags='") << (io.ConfigFlags) << ("'")
                << (std::setw(8)) << (" ImGui::GetIO().ConfigFlags='")
                << (ImGui::GetIO().ConfigFlags) << ("'") << (std::endl)
                << (std::flush);
  }
  {

    std::chrono::duration<double> timestamp =
        std::chrono::high_resolution_clock::now() - g_start_time;
    (std::cout) << (std::setw(10)) << (timestamp.count()) << (" ")
                << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
                << (__LINE__) << (" ") << (__func__) << (" ")
                << ("setup ImGUI style") << (" ") << (std::endl)
                << (std::flush);
  }
  ImGui::StyleColorsDark();
  auto style = ImGui::GetStyle();
  ImGui_ImplGlfw_InitForOpenGL(window.get(), true);
  ImGui_ImplOpenGL3_Init(glsl_version);
  auto font_fn = "/home/martin/src/vcpkg/buildtrees/imgui/src/"
                 "15fb394321-d60a8379fc.clean/misc/fonts/DroidSans.ttf";
  auto font_size = (16.f);
  auto *font = io.Fonts->AddFontFromFileTTF(font_fn, font_size);
  if ((font) == (nullptr)) {
    {

      std::chrono::duration<double> timestamp =
          std::chrono::high_resolution_clock::now() - g_start_time;
      (std::cout) << (std::setw(10)) << (timestamp.count()) << (" ")
                  << (std::this_thread::get_id()) << (" ") << (__FILE__)
                  << (":") << (__LINE__) << (" ") << (__func__) << (" ")
                  << ("loading font failed") << (" ") << (std::setw(8))
                  << (" font_fn='") << (font_fn) << ("'") << (std::setw(8))
                  << (" font_size='") << (font_size) << ("'") << (std::endl)
                  << (std::flush);
    }
  } else {
    {

      std::chrono::duration<double> timestamp =
          std::chrono::high_resolution_clock::now() - g_start_time;
      (std::cout) << (std::setw(10)) << (timestamp.count()) << (" ")
                  << (std::this_thread::get_id()) << (" ") << (__FILE__)
                  << (":") << (__LINE__) << (" ") << (__func__) << (" ")
                  << ("loaded font") << (" ") << (std::setw(8))
                  << (" font_fn='") << (font_fn) << ("'") << (std::setw(8))
                  << (" font_size='") << (font_size) << ("'") << (std::endl)
                  << (std::flush);
    }
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
                (((1.00e+3f)) / (ImGui::GetIO().Framerate)),
                ImGui::GetIO().Framerate);
    ImGui::End();
  }
  fun();
  ImGui::EndFrame();
}
void MainWindow::Render(std::shared_ptr<GLFWwindow> window) {
  auto screen_width = int(0);
  auto screen_height = int(0);
  glfwGetFramebufferSize(window.get(), &screen_width, &screen_height);
  glViewport(0, 0, screen_width, screen_height);
  glClear(GL_COLOR_BUFFER_BIT);
  ImGui::Render();
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
  glfwSwapBuffers(window.get());
}
void MainWindow::Shutdown() {
  {

    std::chrono::duration<double> timestamp =
        std::chrono::high_resolution_clock::now() - g_start_time;
    (std::cout) << (std::setw(10)) << (timestamp.count()) << (" ")
                << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
                << (__LINE__) << (" ") << (__func__) << (" ")
                << ("destroy ImPlot Context") << (" ") << (std::endl)
                << (std::flush);
  }
  ImPlot::DestroyContext();
  {

    std::chrono::duration<double> timestamp =
        std::chrono::high_resolution_clock::now() - g_start_time;
    (std::cout) << (std::setw(10)) << (timestamp.count()) << (" ")
                << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
                << (__LINE__) << (" ") << (__func__) << (" ")
                << ("delete ImGui buffers and textures") << (" ") << (std::endl)
                << (std::flush);
  }
  ImGui_ImplOpenGL3_Shutdown();
  {

    std::chrono::duration<double> timestamp =
        std::chrono::high_resolution_clock::now() - g_start_time;
    (std::cout) << (std::setw(10)) << (timestamp.count()) << (" ")
                << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
                << (__LINE__) << (" ") << (__func__) << (" ")
                << ("delete ImGui callbacks and mouse cursor from GLFW")
                << (" ") << (std::endl) << (std::flush);
  }
  ImGui_ImplGlfw_Shutdown();
  {

    std::chrono::duration<double> timestamp =
        std::chrono::high_resolution_clock::now() - g_start_time;
    (std::cout) << (std::setw(10)) << (timestamp.count()) << (" ")
                << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
                << (__LINE__) << (" ") << (__func__) << (" ")
                << ("destroy ImGui Context") << (" ") << (std::endl)
                << (std::flush);
  }
  ImGui::DestroyContext();
}