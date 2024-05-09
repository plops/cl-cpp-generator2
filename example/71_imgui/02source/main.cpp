#include "Charuco.h"
#include "MainWindow.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_opengl3_loader.h"
#include <GLFW/glfw3.h>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <opencv2/aruco/charuco.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <thread>
std::chrono::time_point<std::chrono::high_resolution_clock> g_start_time;
#include "GraphicsFramework.h"
#include "implot.h"

int main(int argc, char **argv) {
  (g_start_time) = (std::chrono::high_resolution_clock::now());
  {
    // start

    auto framework{GraphicsFramework()};
    auto charuco{Charuco()};
    MainWindow M;
    M.Init(framework.getWindow(), "#version 130");
    charuco.Capture();
    charuco.Init();
    while (!framework.WindowShouldClose()) {
      charuco.Capture();
      framework.PollEvents();
      M.NewFrame();
      M.Update([&charuco]() { charuco.Render(); });
      M.Render(framework.getWindow());
    }
    // run various cleanup functions

    charuco.Shutdown();
    M.Shutdown();
    // leave program

    return 0;
  }
}
