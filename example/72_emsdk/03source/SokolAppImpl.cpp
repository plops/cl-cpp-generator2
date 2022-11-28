// no preamble
;
#include <chrono>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <thread>
extern std::mutex g_stdout_mutex;
extern std::chrono::time_point<std::chrono::high_resolution_clock> g_start_time;
#define SOKOL_GLES2
#define SOKOL_IMPL
#include <sokol_app.h>
#include <sokol_gfx.h>
#include <sokol_glue.h>
#define SOKOL_IMGUI_IMPL
#include "SokolAppImpl.h"
#include <imgui.h>
#include <util/sokol_imgui.h>
// code
;
SokolAppImpl::SokolAppImpl() {
  {

    auto lock = std::unique_lock<std::mutex>(g_stdout_mutex);
    std::chrono::duration<double> timestamp =
        std::chrono::high_resolution_clock::now() - g_start_time;
    (std::cout) << (std::setw(10)) << (timestamp.count()) << (" ")
                << (std::this_thread::get_id()) << (" ") << (__file__) << (":")
                << (__line__) << (" ") << (__func__) << (" ")
                << ("constructor ") << (" ") << (std::endl) << (std::flush);
  }
}
SokolAppImpl::~SokolAppImpl() {
  // destructor
  ;
}
static bool show_test_window = true;
static sg_pass_action pass_action;
extern "C" void init() {
  {

    auto lock = std::unique_lock<std::mutex>(g_stdout_mutex);
    std::chrono::duration<double> timestamp =
        std::chrono::high_resolution_clock::now() - g_start_time;
    (std::cout) << (std::setw(10)) << (timestamp.count()) << (" ")
                << (std::this_thread::get_id()) << (" ") << (__file__) << (":")
                << (__line__) << (" ") << (__func__) << (" ") << ("") << (" ")
                << (std::endl) << (std::flush);
  }
  auto desc = sg_desc();
  desc.context = sapp_sgcontext();
  sg_setup(&desc);
  auto s = simgui_desc_t();
  simgui_setup(&s);
  ImGui::GetIO().ConfigFlags =
      ((ImGui::GetIO().ConfigFlags) | (ImGuiConfigFlags_DockingEnable));
  pass_action.colors[0].action = SG_ACTION_CLEAR;
  pass_action.colors[0].value = {(0.30f), (0.70f), (0.50f), (1.0f)};
}
extern "C" void frame() {
  {
    static int frame_count = 0;
    if ((frame_count % ((10) * (60))) == (0)) {
      {

        auto lock = std::unique_lock<std::mutex>(g_stdout_mutex);
        std::chrono::duration<double> timestamp =
            std::chrono::high_resolution_clock::now() - g_start_time;
        (std::cout) << (std::setw(10)) << (timestamp.count()) << (" ")
                    << (std::this_thread::get_id()) << (" ") << (__file__)
                    << (":") << (__line__) << (" ") << (__func__) << (" ")
                    << ("") << (" ") << (std::endl) << (std::flush);
      }
    }
    (frame_count)++;
  }
  auto w = sapp_width();
  auto h = sapp_height();
  simgui_new_frame({w, h, sapp_frame_duration(), sapp_dpi_scale()});
  {
    static float f = 0.0f;
    ImGui::Text("drag windows");
    ImGui::SliderFloat("float", &f, (0.f), (1.0f));
    if (ImGui::Button("window")) {
      show_test_window = !show_test_window;
    }
  }
  sg_begin_default_pass(&pass_action, w, h);
  simgui_render();
  sg_end_pass();
  sg_commit();
}
extern "C" void cleanup() {
  {

    auto lock = std::unique_lock<std::mutex>(g_stdout_mutex);
    std::chrono::duration<double> timestamp =
        std::chrono::high_resolution_clock::now() - g_start_time;
    (std::cout) << (std::setw(10)) << (timestamp.count()) << (" ")
                << (std::this_thread::get_id()) << (" ") << (__file__) << (":")
                << (__line__) << (" ") << (__func__) << (" ") << ("") << (" ")
                << (std::endl) << (std::flush);
  }
  simgui_shutdown();
  sg_shutdown();
}
extern "C" void input(const sapp_event *event) {
  {

    auto lock = std::unique_lock<std::mutex>(g_stdout_mutex);
    std::chrono::duration<double> timestamp =
        std::chrono::high_resolution_clock::now() - g_start_time;
    (std::cout) << (std::setw(10)) << (timestamp.count()) << (" ")
                << (std::this_thread::get_id()) << (" ") << (__file__) << (":")
                << (__line__) << (" ") << (__func__) << (" ") << ("") << (" ")
                << (std::endl) << (std::flush);
  }
  simgui_handle_event(event);
}
extern "C" sapp_desc sokol_main(int argc, char **argv) {
  g_start_time = std::chrono::high_resolution_clock::now();
  {

    auto lock = std::unique_lock<std::mutex>(g_stdout_mutex);
    std::chrono::duration<double> timestamp =
        std::chrono::high_resolution_clock::now() - g_start_time;
    (std::cout) << (std::setw(10)) << (timestamp.count()) << (" ")
                << (std::this_thread::get_id()) << (" ") << (__file__) << (":")
                << (__line__) << (" ") << (__func__) << (" ")
                << ("enter program") << (" ") << (std::setw(8)) << (" argc='")
                << (argc) << ("'") << (std::setw(8)) << (" argv='") << (argv)
                << ("'") << (std::endl) << (std::flush);
  }
  auto s = sapp_desc();
  s.width = 640;
  s.height = 480;
  s.init_cb = init;
  s.frame_cb = frame;
  s.cleanup_cb = cleanup;
  s.event_cb = input;
  s.gl_force_gles2 = true;
  s.window_title = "imgui docking";
  s.ios_keyboard_resizes_canvas = false;
  s.icon.sokol_default = true;
  {

    auto lock = std::unique_lock<std::mutex>(g_stdout_mutex);
    std::chrono::duration<double> timestamp =
        std::chrono::high_resolution_clock::now() - g_start_time;
    (std::cout) << (std::setw(10)) << (timestamp.count()) << (" ")
                << (std::this_thread::get_id()) << (" ") << (__file__) << (":")
                << (__line__) << (" ") << (__func__) << (" ")
                << ("exit program") << (" ") << (std::endl) << (std::flush);
  }
  return s;
}