#include "CpuAffinityManagerWithGui.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "implot.h"
#include <GLFW/glfw3.h>
#include <chrono>
#include <cmath>
#include <deque>
#include <format>
#include <iostream>
#include <unistd.h>
#include <vector>
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

std::tuple<system_info, unsigned char *, pm_table> start_pm_monitor2() {
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
  if (0x400005 == obj.pm_table_version) {
    if (SMU_Return_OK == smu_read_pm_table(&obj, pm_buf, obj.pm_table_size)) {
      std::cout << std::format("PMT hack for cezanne's core_disabled_map\n");
      disabled_cores_0x400005(&pmt, &sysinfo);
    }
  }
  get_processor_topology(&sysinfo, pmt.zen_version);
  switch (obj.smu_if_version) {
  case IF_VERSION_9: {
    sysinfo.if_ver = 9;
    break;
  };
  case IF_VERSION_10: {
    sysinfo.if_ver = 10;
    break;
  };
  case IF_VERSION_11: {
    sysinfo.if_ver = 11;
    break;
  };
  case IF_VERSION_12: {
    sysinfo.if_ver = 12;
    break;
  };
  case IF_VERSION_13: {
    sysinfo.if_ver = 13;
    break;
  };
  default: {
    sysinfo.if_ver = 0;
    break;
  };
  }
  return std::make_tuple(sysinfo, pm_buf, pmt);
}

#define pmta(elem) ((pmt.elem) ? (*(pmt.elem)) : std::nanf("1"))
#define pmta0(elem) ((pmt.elem) ? (*(pmt.elem)) : 0F)

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
  auto vsyncOn{true};
  glfwSwapInterval(vsyncOn ? 1 : 0);
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImPlot::CreateContext();
  auto io{ImGui::GetIO()};
  io.ConfigFlags = io.ConfigFlags | ImGuiConfigFlags_NavEnableKeyboard;
  ImGui::StyleColorsDark();
  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init(glsl_version);
  auto [sysinfo, pm_buf, pmt]{start_pm_monitor2()};
  auto show_demo_window{true};
  auto clear_color{ImVec4(0.40F, 0.50F, 0.60F, 1.0F)};
  auto maxDataPoints{1024};
  auto startTime{std::chrono::steady_clock::now()};
  auto diagramVoltage{DiagramWithGui(8, maxDataPoints, "voltage")};
  auto diagramTemperature{DiagramWithGui(8, maxDataPoints, "temperature")};
  auto diagramFrequency{DiagramWithGui(8, maxDataPoints, "frequency")};
  auto diagramPower{DiagramWithGui(8, maxDataPoints, "power")};
  auto affinityManager{CpuAffinityManagerWithGui(getpid())};
  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    if (show_demo_window) {
      ImGui::ShowDemoWindow(&show_demo_window);
      ImPlot::ShowDemoWindow();
    }
    if (SMU_Return_OK == smu_read_pm_table(&obj, pm_buf, obj.pm_table_size)) {
      if (sysinfo.available) {
        ImGui::Begin("Ryzen");
        affinityManager.RenderGui();
        if (ImGui::Checkbox("vsync", &vsyncOn)) {
          glfwSwapInterval(vsyncOn ? 1 : 0);
        }
        ImGui::Text("%s",
                    std::format("cpu_name='{}'", sysinfo.cpu_name).c_str());
        ImGui::Text("%s",
                    std::format("codename='{}'", sysinfo.codename).c_str());
        ImGui::Text("%s", std::format("cores='{}'", sysinfo.cores).c_str());
        ImGui::Text("%s", std::format("ccds='{}'", sysinfo.ccds).c_str());
        ImGui::Text("%s", std::format("ccxs='{}'", sysinfo.ccxs).c_str());
        ImGui::Text(
            "%s",
            std::format("cores_per_ccx='{}'", sysinfo.cores_per_ccx).c_str());
        ImGui::Text("%s",
                    std::format("smu_fw_ver='{}'", sysinfo.smu_fw_ver).c_str());
        ImGui::Text("%s", std::format("if_ver='{}'", sysinfo.if_ver).c_str());
        auto package_sleep_time{0.F};
        auto average_voltage{0.F};
        if (pmt.PC6) {
          package_sleep_time = (pmta(PC6) / 1.00e+2F);
          average_voltage =
              ((pmta(CPU_TELEMETRY_VOLTAGE) - (0.20F * package_sleep_time)) /
               (1.0F - package_sleep_time));
        } else {
          average_voltage = pmta(CPU_TELEMETRY_VOLTAGE);
        }
        auto currentTime{std::chrono::steady_clock::now()};
        auto elapsedTime{
            std::chrono::duration<float>(currentTime - startTime).count()};
        auto voltageValues{std::vector<float>(pmt.max_cores)};
        auto temperatureValues{std::vector<float>(pmt.max_cores)};
        auto frequencyValues{std::vector<float>(pmt.max_cores)};
        auto powerValues{std::vector<float>(pmt.max_cores)};
        for (auto i = 0; i < pmt.max_cores; i += 1) {
          auto core_disabled{sysinfo.core_disable_map >> i & 1};
          auto core_frequency{pmta(CORE_FREQEFF[i]) * 1.00e+3F};
          auto core_voltage_true{pmta(CORE_VOLTAGE[i])};
          auto core_sleep_time{pmta(CORE_CC6[i]) / 1.00e+2F};
          auto core_voltage{(1.0F - core_sleep_time) * average_voltage +
                            0.20F * core_sleep_time};
          auto core_temperature{pmta(CORE_TEMP[i])};
          auto core_power{pmta(CORE_POWER[i])};
          voltageValues[i] = core_voltage;
          temperatureValues[i] = core_temperature;
          frequencyValues[i] = core_frequency;
          powerValues[i] = core_power;
          if (core_disabled) {
            ImGui::Text("%s", std::format("{:2} Disabled", i).c_str());
          } else {
            ImGui::Text(
                "%s",
                std::format("{:2} {} {:6.3f}W {:5.3f}V {:5.3f}V {:6.2f}C C0: "
                            "{:5.1f}% C1: {:5.1f}% C6: {:5.1f}%",
                            i,
                            6.0F <= pmta(CORE_C0[i])
                                ? "Sleeping  "
                                : std::format("{:7.1f}MHz", core_frequency),
                            core_power, core_voltage, core_voltage_true,
                            core_temperature, pmta(CORE_C0[i]),
                            pmta(CORE_CC1[i]), pmta(CORE_CC6[i]))
                    .c_str());
          }
        }
        diagramVoltage.AddDataPoint(elapsedTime, voltageValues);
        diagramFrequency.AddDataPoint(elapsedTime, frequencyValues);
        diagramTemperature.AddDataPoint(elapsedTime, temperatureValues);
        diagramPower.AddDataPoint(elapsedTime, powerValues);
        diagramTemperature.RenderGui();
        diagramPower.RenderGui();
        diagramFrequency.RenderGui();
        diagramVoltage.RenderGui();
        ImGui::End();
      }
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
