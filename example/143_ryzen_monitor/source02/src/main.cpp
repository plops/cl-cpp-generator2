#include "CpuAffinityManagerWithGui.h"
#include "DiagramWithGui.h"
#include "imgui.h"
#include "imgui_impl_sdl2.h"
#include "imgui_impl_opengl3.h"
#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>
#include "FancyWindow.h"
#include "implot.h"
#include <chrono>
#include <cmath>
#include <deque>
#include <format>
#include <iostream>
#include <popl.hpp>
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
    auto op{popl::OptionParser("allowed options")};
    auto maxThreads{int(12)};
    auto maxDataPoints{int(1024)};
    auto helpOption{op.add<popl::Switch>("h", "help", "produce help message")};
    auto verboseOption{
            op.add<popl::Switch>("v", "verbose", "produce verbose output")};
    auto maxThreadsOption{op.add<popl::Value<int>>("t", "maxThreads", "parameter",
                                                   12, &maxThreads)};
    auto maxDataPointsOption{op.add<popl::Value<int>>(
            "n", "maxDataPoints", "parameter", 1024, &maxDataPoints)};
    op.parse(argc, argv);
    if (helpOption->count()) {
        std::cout << op << std::endl;
        exit(0);
    }
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

    auto glsl_version{"#version 130"};
    auto w = FancyWindow({.Width = 320, .Height = 240});
    w.updateFrom();
    //while (true) {
     //   w.present();
      //  SDL_Delay(15);
    //}

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();
    auto io{ImGui::GetIO()};
    io.ConfigFlags = io.ConfigFlags | ImGuiConfigFlags_NavEnableKeyboard;
    ImGui::StyleColorsDark();


    SDL_GLContext gl_context = SDL_GL_CreateContext(w.Window_);
    SDL_GL_MakeCurrent(w.Window_, gl_context);
    SDL_GL_SetSwapInterval(1); // Enable vsync


    ImGui_ImplSDL2_InitForOpenGL(w.Window_, gl_context);

    ImGui_ImplOpenGL3_Init(glsl_version);
    auto [sysinfo, pm_buf, pmt]{start_pm_monitor2()};
    auto clear_color{ImVec4(0.40F, 0.50F, 0.60F, 1.0F)};
    auto startTime{std::chrono::steady_clock::now()};
    auto temperatureDiagram{DiagramWithGui(8, maxDataPoints, "temperature")};
    auto powerDiagram{DiagramWithGui(8, maxDataPoints, "power")};
    auto frequencyDiagram{DiagramWithGui(8, maxDataPoints, "frequency")};
    auto voltageDiagram{DiagramWithGui(8, maxDataPoints, "voltage")};
    auto c0Diagram{DiagramWithGui(8, maxDataPoints, "c0")};
    auto cc1Diagram{DiagramWithGui(8, maxDataPoints, "cc1")};
    auto cc6Diagram{DiagramWithGui(8, maxDataPoints, "cc6")};
    auto power_memDiagram{DiagramWithGui(2, maxDataPoints, "power_mem")};
    auto power_socketDiagram{DiagramWithGui(1, maxDataPoints, "power_socket")};
    auto affinityManager{CpuAffinityManagerWithGui(getpid(), maxThreads)};
    bool sumOn{false};
    while (true) {
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplSDL2_NewFrame();
        ImGui::NewFrame();
        if (SMU_Return_OK == smu_read_pm_table(&obj, pm_buf, obj.pm_table_size)) {
            if (sysinfo.available) {
                ImGui::Begin("Ryzen");
                affinityManager.RenderGui();

                ImGui::Checkbox("sum", &sumOn);
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
                auto temperatureValues{std::vector<float>(pmt.max_cores)};
                auto powerValues{std::vector<float>(pmt.max_cores)};
                auto frequencyValues{std::vector<float>(pmt.max_cores)};
                auto voltageValues{std::vector<float>(pmt.max_cores)};
                auto c0Values{std::vector<float>(pmt.max_cores)};
                auto cc1Values{std::vector<float>(pmt.max_cores)};
                auto cc6Values{std::vector<float>(pmt.max_cores)};
                auto l3_logic_power{pmta(L3_LOGIC_POWER[0])};
                auto ddr_phy_power{pmta(DDR_PHY_POWER)};
                auto socket_power{pmta(SOCKET_POWER)};
                ImGui::Text("%s",
                            std::format("L3={:6.3f}W DDR={:6.3f}W SOCKET={:6.3f}W",
                                        l3_logic_power, ddr_phy_power, socket_power)
                                    .c_str());
                for (decltype(0 + pmt.max_cores + 1) i = 0; i < pmt.max_cores; i += 1) {
                    auto core_disabled{sysinfo.core_disable_map >> i & 1};
                    auto core_frequency{pmta(CORE_FREQEFF[i]) * 1.00e+3F};
                    auto core_voltage_true{pmta(CORE_VOLTAGE[i])};
                    auto core_sleep_time{pmta(CORE_CC6[i]) / 1.00e+2F};
                    auto core_voltage{(1.0F - core_sleep_time) * average_voltage +
                                      0.20F * core_sleep_time};
                    auto core_temperature{pmta(CORE_TEMP[i])};
                    auto core_power{pmta(CORE_POWER[i])};
                    auto core_c0{pmta(CORE_C0[i])};
                    auto core_cc1{pmta(CORE_CC1[i])};
                    auto core_cc6{pmta(CORE_CC6[i])};
                    temperatureValues[i] = core_temperature;
                    powerValues[i] = core_power;
                    frequencyValues[i] = core_frequency;
                    voltageValues[i] = core_voltage;
                    c0Values[i] = core_c0;
                    cc1Values[i] = core_cc1;
                    cc6Values[i] = core_cc6;
                    if (core_disabled) {
                        ImGui::Text(
                                "%s",
                                std::format(
                                        "{:2} Disabled   {:6.3f}W {:5.3f}V {:5.3f}V {:6.2f}C ", i,
                                        core_power, core_voltage, core_voltage_true,
                                        core_temperature)
                                        .c_str());
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
                                            core_temperature, core_c0, core_cc1, core_cc6)
                                        .c_str());
                    }
                }
                temperatureDiagram.AddDataPoint(elapsedTime, temperatureValues);
                powerDiagram.AddDataPoint(elapsedTime, powerValues);
                frequencyDiagram.AddDataPoint(elapsedTime, frequencyValues);
                voltageDiagram.AddDataPoint(elapsedTime, voltageValues);
                c0Diagram.AddDataPoint(elapsedTime, c0Values);
                cc1Diagram.AddDataPoint(elapsedTime, cc1Values);
                cc6Diagram.AddDataPoint(elapsedTime, cc6Values);
                power_memDiagram.AddDataPoint(
                        elapsedTime, std::vector<float>({l3_logic_power, ddr_phy_power}));
                power_socketDiagram.AddDataPoint(elapsedTime,
                                                 std::vector<float>({socket_power}));
                temperatureDiagram.RenderGui(false);
                if (sumOn) {
                    powerDiagram.RenderGuiSum(false);
                } else {
                    powerDiagram.RenderGui(false);
                }
                frequencyDiagram.RenderGui(false);
                voltageDiagram.RenderGui(false);
                if (sumOn) {
                    c0Diagram.RenderGuiSum(false);
                } else {
                    c0Diagram.RenderGui(false);
                }
                if (sumOn) {
                    cc1Diagram.RenderGuiSum(false);
                } else {
                    cc1Diagram.RenderGui(false);
                }
                if (sumOn) {
                    cc6Diagram.RenderGuiSum(false);
                } else {
                    cc6Diagram.RenderGui(false);
                }
                power_memDiagram.RenderGui(false);
                power_socketDiagram.RenderGui(true);
                ImGui::End();
            }
        }
        ImGui::Render();
//        auto wi{0};
//        auto h{0};
//
//        //glfwGetFramebufferSize(w.Window_, &wi, &h);
//        glViewport(0, 0, wi, h);
        glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w,
                     clear_color.z * clear_color.w, clear_color.w);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        w.present();
        SDL_GL_SwapWindow(w.Window_);
    }
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();
    //glfwDestroyWindow(window);
    //glfwTerminate();
    return 0;
}
