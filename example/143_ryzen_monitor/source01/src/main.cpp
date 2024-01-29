#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "implot.h"
#include <GLFW/glfw3.h>
#include <format>
#include <iostream>
#include <unistd.h>
extern "C" {
#include <libsmu.h>
#include <pm_tables.h>
#include <readinfo.h>
extern smu_obj_t obj;
void start_pm_monitor(unsigned int);
int select_pm_table_version(unsigned int version, pm_table *pmt,
                            unsigned char *pm_buf);
void disabled_cores_0x400005(pm_table *pmt, system_info *sysinfo);
};

void glfw_error_callback(int err, const char *description) {
  std::cout << std::format(" err='{}' description='{}'\n", err, description);
}

void start_pm_monitor2() {
  if (!smu_pm_tables_supported(&obj)) {
    std::cout << std::format("pm tables not supported on this platform\n");
    exit(0);
  }
  auto pm_buf{static_cast<unsigned char *>(
      calloc(obj.pm_table_size, sizeof(unsigned char)))};
  if (!pm_buf) {
    std::cout << std::format("could not allocate PM Table\n");
    exit(0);
  }
  auto pmt{pm_table()};
  if (!select_pm_table_version(obj.pm_table_version, &pmt, pm_buf)) {
    std::cout << std::format("pm table version not supported\n");
    exit(0);
  }
  if (obj.pm_table_size < pmt.min_size) {
    std::cout << std::format(
        "pm table larger than the table returned by smu\n");
    exit(0);
  }
  auto sysinfo{system_info()};
  sysinfo.enabled_cores_count = pmt.max_cores;
  sysinfo.cpu_name = get_processor_name();
  sysinfo.codename = smu_codename_to_str(&obj);
  sysinfo.smu_fw_ver = smu_get_fw_version(&obj);
}

int main(int argc, char **argv) {
  if (0 != getuid() && 0 != geteuid()) {
    std::cout << std::format("Program must be run as root\n");
    return 1;
  }
  auto ret{static_cast<smu_return_val>(smu_init(&obj))};
  if (!(SMU_Return_OK == ret)) {
    std::cout << std::format("error smu_return_to_str(ret)='{}'\n",
                             smu_return_to_str(ret));
    return 1;
  }
  start_pm_monitor2();
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
