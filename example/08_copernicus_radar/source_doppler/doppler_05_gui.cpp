
#include "utils.h"

#include "globals.h"

#include "proto2.h"
;
extern State state;
// https://youtu.be/nVaQuNXueFw?t=317
#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl2.h"
void initGui() {
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
void drawGui() {
  ImGui_ImplOpenGL2_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();
  static int slider_a = 50;
  ImGui::SliderInt("slider_a", &(slider_a), 0, 99);
  state._range_line = runProcessing(slider_a);
  {
    // plot raw data (real)
    auto p = reinterpret_cast<std::complex<float> *>(state._mmap_data);
    auto range = state._range;
    auto h_signal = &(p[((range) * (slider_a))]);
    static float *range_raw_re =
        static_cast<float *>(malloc(((sizeof(float)) * (range))));
    for (int i = 0; i < range; (i) += (1)) {
      range_raw_re[i] = std::real(h_signal[i]);
    }
    if (range_raw_re) {
      ImGui::PlotLines("range_raw_re", range_raw_re, range, 0, NULL, FLT_MAX,
                       FLT_MAX, ImVec2(3700, 400), sizeof(float));
    };
  };
  auto range = state._range;
  static float *range_abs =
      static_cast<float *>(malloc(((sizeof(float)) * (range))));
  static float *range_re =
      static_cast<float *>(malloc(((sizeof(float)) * (range))));
  if (state._range_line) {
    for (int i = 0; i < range; (i) += (1)) {
      range_abs[i] = std::abs(state._range_line[i]);
      range_re[i] = std::real(state._range_line[i]);
    }
    ImGui::PlotLines("range_abs", range_abs, range, 0, NULL, FLT_MAX, FLT_MAX,
                     ImVec2(3200, 400), sizeof(float));
    ImGui::PlotLines("range_re", range_re, range, 0, NULL, FLT_MAX, FLT_MAX,
                     ImVec2(3200, 400), sizeof(float));
  };
  ImGui::Render();
  ImGui_ImplOpenGL2_RenderDrawData(ImGui::GetDrawData());
};