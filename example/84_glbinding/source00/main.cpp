#include <chrono>
#include <glbinding/AbstractFunction.h>
#include <glbinding/CallbackMask.h>
#include <glbinding/FunctionCall.h>
#include <glbinding/gl32core/gl.h>
#include <glbinding/glbinding.h>
#include <iomanip>
#include <iostream>
#include <thread>
using namespace gl32core;
using namespace glbinding;
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>
#include <imgui.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <imgui_entt_entity_editor.hpp>
const std::chrono::time_point<std::chrono::high_resolution_clock> g_start_time =
    std::chrono::high_resolution_clock::now();
class Transform {
public:
  float x = (0.f);
  float y = (0.f);
};
class Velocity {
public:
  float x = (0.f);
  float y = (0.f);
};
void computeVelocity(entt::registry &reg, float delta, float width,
                     float height, float radius) {
  reg.view<Transform, Velocity>().each([&](Transform &trans, Velocity &vel) {
    (trans.x) += (((vel.x) * (delta)));
    (trans.y) += (((vel.y) * (delta)));
    if ((((trans.x) < (radius)) || ((((width) - (radius))) < (trans.x)))) {
      trans.x = std::clamp(trans.x, radius, ((width) - (radius)));
      vel.x = -vel.x;
    }
    if ((((trans.y) < (radius)) || ((((height) - (radius))) < (trans.y)))) {
      trans.y = std::clamp(trans.y, radius, ((height) - (radius)));
      vel.y = -vel.y;
    }
  });
}
namespace MM {
template <>
void ComponentEditorWidget<Transform>(entt::registry &reg,
                                      entt::registry::entity_type e) {
  auto &t = reg.get<Transform>(e);
  const auto step = (0.10f);
  ImGui::DragFloat("x", &t.x, step);
  ImGui::DragFloat("y", &t.y, step);
};
template <>
void ComponentEditorWidget<Velocity>(entt::registry &reg,
                                     entt::registry::entity_type e) {
  auto &t = reg.get<Velocity>(e);
  const auto step = (0.10f);
  ImGui::DragFloat("x", &t.x, step);
  ImGui::DragFloat("y", &t.y, step);
};
}; // namespace MM
void lprint(std::initializer_list<std::string> il) {
  std::chrono::duration<double> timestamp(0);
  timestamp = ((std::chrono::high_resolution_clock::now()) - (g_start_time));
  const auto defaultWidth = 10;
  (std::cout) << (std::setw(defaultWidth)) << (timestamp.count()) << (" ")
              << (std::this_thread::get_id()) << (" ");
  for (const auto &elem : il) {
    (std::cout) << (elem);
  }
  (std::cout) << (std::endl) << (std::flush);
}
int main(int argc, char **argv) {
  lprint({std::to_string(__LINE__), " ", &(__PRETTY_FUNCTION__[0]), " ",
          "start", " ", " argc='", std::to_string(argc), "'"});
  auto *window = ([]() -> GLFWwindow * {
    lprint({std::to_string(__LINE__), " ", &(__PRETTY_FUNCTION__[0]), " ",
            "initialize GLFW3", " "});
    if (!(glfwInit())) {
      lprint({std::to_string(__LINE__), " ", &(__PRETTY_FUNCTION__[0]), " ",
              "glfwInit failed", " "});
    }
    glfwWindowHint(GLFW_VISIBLE, true);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, true);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    // enable Vsync
    ;
    glfwSwapInterval(1);
    lprint({std::to_string(__LINE__), " ", &(__PRETTY_FUNCTION__[0]), " ",
            "create GLFW3 window", " "});
    const auto startWidth = 800;
    const auto startHeight = 600;
    auto window = glfwCreateWindow(startWidth, startHeight, "hello bgfx",
                                   nullptr, nullptr);
    if (!(window)) {
      lprint({std::to_string(__LINE__), " ", &(__PRETTY_FUNCTION__[0]), " ",
              "can't create glfw window", " "});
    }
    lprint({std::to_string(__LINE__), " ", &(__PRETTY_FUNCTION__[0]), " ",
            "initialize GLFW3 context for window", " "});
    glfwMakeContextCurrent(window);
    return window;
  })();
  auto width = int(0);
  auto height = int(0);
  lprint({std::to_string(__LINE__), " ", &(__PRETTY_FUNCTION__[0]), " ",
          "initialize glbinding", " "});
  // if second arg is false: lazy function pointer loading
  ;
  glbinding::initialize(glfwGetProcAddress, false);
  {
    const float r = (0.40f);
    const float g = (0.40f);
    const float b = (0.20f);
    const float a = (1.0f);
    glClearColor(r, g, b, a);
  }
  lprint({std::to_string(__LINE__), " ", &(__PRETTY_FUNCTION__[0]), " ",
          "initialize ImGui", " "});
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  auto io = ImGui::GetIO();
  io.ConfigFlags = ((io.ConfigFlags) | (ImGuiConfigFlags_NavEnableKeyboard));
  ImGui::StyleColorsLight();
  const auto installCallbacks = true;
  ImGui_ImplGlfw_InitForOpenGL(window, installCallbacks);
  const auto glslVersion = "#version 150";
  ImGui_ImplOpenGL3_Init(glslVersion);
  lprint({std::to_string(__LINE__), " ", &(__PRETTY_FUNCTION__[0]), " ",
          "initialize ENTT", " "});
  entt::registry reg;
  MM::EntityEditor<entt::entity> editor;
  editor.registerComponent<Transform>("Transform");
  editor.registerComponent<Velocity>("Velocity");
  const auto n = 1000;
  for (auto i = 0; (i) < (n); (i) += (1)) {
    auto e = reg.create();
    const auto range = 5000;
    const auto offset = ((range) / (2));
    const auto scale = (0.10f);
    reg.emplace<Transform>(e, ((scale) * (static_cast<float>(rand() % range))),
                           ((scale) * (static_cast<float>(rand() % range))));
    reg.emplace<Velocity>(
        e, ((scale) * (static_cast<float>(((-offset) + (rand() % range))))),
        ((scale) * (static_cast<float>(((-offset) + (rand() % range))))));
  }
  const auto radius = (10.f);
  lprint({std::to_string(__LINE__), " ", &(__PRETTY_FUNCTION__[0]), " ",
          "start loop", " "});
  while (!(glfwWindowShouldClose(window))) {
    glfwPollEvents();
    const auto framesPerSecond = (60.f);
    computeVelocity(reg, (((1.0f)) / (framesPerSecond)),
                    static_cast<float>(width), static_cast<float>(height),
                    radius);
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    {
      ImGui::NewFrame();
      auto *dl = ImGui::GetBackgroundDrawList();
      reg.view<Transform>().each([&](auto e, Transform &trans) {
        auto eInt = ((1) + (entt::to_integral(e)));
        const auto M = 256;
        const auto R = 13;
        const auto G = 159;
        const auto B = 207;
        const auto A = 250;
        const auto colorBasedOnId = IM_COL32(
            ((R) * (eInt)) % M, ((G) * (eInt)) % M, ((B) * (eInt)) % M, A);
        dl->AddCircleFilled(ImVec2(trans.x, trans.y), radius, colorBasedOnId);
      });
      auto showDemoWindow = true;
      ImGui::ShowDemoWindow(&showDemoWindow);
      ImGui::Render();
    }
    ([&width, &height, window]() {
      // react to changing window size
      ;
      auto oldwidth = width;
      auto oldheight = height;
      glfwGetWindowSize(window, &width, &height);
      if ((((width) != (oldwidth)) || ((height) != (oldheight)))) {
        lprint({std::to_string(__LINE__), " ", &(__PRETTY_FUNCTION__[0]), " ",
                "window size has changed", " ", " width='",
                std::to_string(width), "'", " height='", std::to_string(height),
                "'"});
        glViewport(0, 0, width, height);
      }
    })();
    // draw frame
    ;
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    glfwSwapBuffers(window);
  }
  lprint({std::to_string(__LINE__), " ", &(__PRETTY_FUNCTION__[0]), " ",
          "Shutdown ImGui and GLFW3", " "});
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
  glfwDestroyWindow(window);
  glfwTerminate();
  return 0;
}