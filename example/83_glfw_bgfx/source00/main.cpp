#include <GLFW/glfw3.h>
#include <bgfx/bgfx.h>
#include <bgfx/platform.h>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <thread>
#define GLFW_EXPOSE_NATIVE_X11
#include <GLFW/glfw3native.h>
#include <imgui/imgui.h>
std::chrono::time_point<std::chrono::high_resolution_clock> g_start_time;
void lprint(std::initializer_list<std::string> il) {
  std::chrono::duration<double> timestamp =
      std::chrono::high_resolution_clock::now() - g_start_time;
  (std::cout) << (std::setw(10)) << (timestamp.count()) << (" ")
              << (std::this_thread::get_id()) << (" ");
  for (const auto &elem : il) {
    (std::cout) << (elem);
  }
  (std::cout) << (std::endl) << (std::flush);
}
int main(int argc, char **argv) {
  g_start_time = std::chrono::high_resolution_clock::now();
  lprint({__FILE__, ":", std::to_string(__LINE__), " ", __func__, " ", "start",
          " ", " argc='", std::to_string(argc), "'"});
  auto *window = ([]() -> GLFWwindow * {
    if (!(glfwInit())) {
      lprint({__FILE__, ":", std::to_string(__LINE__), " ", __func__, " ",
              "glfwInit failed", " "});
    }
    auto window = glfwCreateWindow(800, 600, "hello bgfx", nullptr, nullptr);
    if (!(window)) {
      lprint({__FILE__, ":", std::to_string(__LINE__), " ", __func__, " ",
              "can't create glfw window", " "});
    }
    return window;
  })();
  auto width = int(0);
  auto height = int(0);
  ([&width, &height, window]() {
    // call renderFrame before bgfx::init to signal to bgfx not to create a
    // render thread
    ;
    bgfx::renderFrame();
    auto bi = bgfx::Init();
    bi.platformData.ndt = glfwGetX11Display();
    bi.platformData.nwh = reinterpret_cast<void *>(
        static_cast<uintptr_t>(glfwGetX11Window(window)));
    glfwGetWindowSize(window, &width, &height);
    bi.type = bgfx::RendererType::Count;
    bi.resolution.width = width;
    bi.resolution.height = height;
    bi.resolution.reset = BGFX_RESET_VSYNC;
    if (!(bgfx::init(bi))) {
      lprint({__FILE__, ":", std::to_string(__LINE__), " ", __func__, " ",
              "bgfx init failed", " "});
    }
    auto debug = BGFX_DEBUG_TEXT;
    bgfx::setDebug(debug);
    bgfx::setViewClear(0, ((BGFX_CLEAR_COLOR) | (BGFX_CLEAR_DEPTH)), 0x303030FF,
                       (1.0f), 0);
    bgfx::setViewRect(0, 0, 0, bgfx::BackbufferRatio::Equal);
    imguiCreate();
  })();
  while (!(glfwWindowShouldClose(window))) {
    glfwPollEvents();
    ([&width, &height, window]() {
      auto oldwidth = width;
      auto oldheight = height;
      glfwGetWindowSize(window, &width, &height);
      if ((((width) != (oldwidth)) || ((height) != (oldheight)))) {
        bgfx::reset(width, height, BGFX_RESET_VSYNC);
        bgfx::setViewRect(0, 0, 0, bgfx::BackbufferRatio::Equal);
      }
    })();
    bgfx::touch(0);
    bgfx::dbgTextClear();
    bgfx::dbgTextPrintf(0, 0, 15, "press F1 to toggle stats");
    bgfx::setDebug(BGFX_DEBUG_STATS);
    bgfx::frame();
  }
  bgfx::shutdown();
  glfwTerminate();
  return 0;
}