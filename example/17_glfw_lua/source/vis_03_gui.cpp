
#include "utils.h"

#include "globals.h"

#include "proto2.h"
;
extern State state;
// https://youtu.be/nVaQuNXueFw?t=317
// https://blog.conan.io/2019/06/26/An-introduction-to-the-Dear-ImGui-library.html
#include "imgui/examples/imgui_impl_glfw.h"
#include "imgui/examples/imgui_impl_opengl2.h"
#include "imgui/imgui.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
void initGui() {

  (std::cout)
      << (std::setw(10))
      << (std::chrono::high_resolution_clock::now().time_since_epoch().count())
      << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
      << (__LINE__) << (" ") << (__func__) << (" ") << ("initGui") << (" ")
      << (std::endl) << (std::flush);
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGui_ImplGlfw_InitForOpenGL(state._window, true);
  ImGui_ImplOpenGL2_Init();
  ImGui::StyleColorsDark();
}
void cleanupGui() {
  ImGui_ImplOpenGL2_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
}
float get_FixedDeque(void *data, int idx) {
  auto data1 = reinterpret_cast<FixedDeque<120> *>(data);
  return data1[0][idx];
}
void drawGui() {
  ImGui_ImplOpenGL2_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();
  ImGui::Begin("snapped_cursor");
  ImGui::Text("x: %04d y: %04d",
              static_cast<int>(state._snapped_world_cursor[0]),
              static_cast<int>(state._snapped_world_cursor[1]));
  ImGui::End();
  auto b = true;
  ImGui::ShowDemoWindow(&b);
  ImGui::Render();
  ImGui_ImplOpenGL2_RenderDrawData(ImGui::GetDrawData());
};