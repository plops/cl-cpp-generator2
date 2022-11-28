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
  g_start_time = std::chrono::high_resolution_clock::now();
  {
    {

      std::chrono::duration<double> timestamp =
          std::chrono::high_resolution_clock::now() - g_start_time;
      (std::cout) << (std::setw(10)) << (timestamp.count()) << (" ")
                  << (std::this_thread::get_id()) << (" ") << (__FILE__)
                  << (":") << (__LINE__) << (" ") << (__func__) << (" ")
                  << ("start") << (" ") << (std::setw(8)) << (" argc='")
                  << (argc) << ("'") << (std::setw(8)) << (" argv[0]='")
                  << (argv[0]) << ("'") << (std::endl) << (std::flush);
    }
    auto framework = GraphicsFramework();
    auto charuco = Charuco();
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
    {

      std::chrono::duration<double> timestamp =
          std::chrono::high_resolution_clock::now() - g_start_time;
      (std::cout) << (std::setw(10)) << (timestamp.count()) << (" ")
                  << (std::this_thread::get_id()) << (" ") << (__FILE__)
                  << (":") << (__LINE__) << (" ") << (__func__) << (" ")
                  << ("run various cleanup functions") << (" ") << (std::endl)
                  << (std::flush);
    }
    charuco.Shutdown();
    M.Shutdown();
    {

      std::chrono::duration<double> timestamp =
          std::chrono::high_resolution_clock::now() - g_start_time;
      (std::cout) << (std::setw(10)) << (timestamp.count()) << (" ")
                  << (std::this_thread::get_id()) << (" ") << (__FILE__)
                  << (":") << (__LINE__) << (" ") << (__func__) << (" ")
                  << ("leave program") << (" ") << (std::endl) << (std::flush);
    }
    return 0;
  }
}