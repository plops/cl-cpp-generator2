#include "MainWindow.h"
#include "SysInfo.h"
#include <QApplication>
#include <QMainWindow>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <thread>
std::chrono::time_point<std::chrono::high_resolution_clock> g_start_time;
int main(int argc, char **argv) {
  g_start_time = std::chrono::high_resolution_clock::now();
  {

    std::chrono::duration<double> timestamp =
        std::chrono::high_resolution_clock::now() - g_start_time;
    (std::cout) << (std::setw(10)) << (timestamp.count()) << (" ")
                << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
                << (__LINE__) << (" ") << (__func__) << (" ") << ("") << (" ")
                << (std::endl) << (std::flush);
  }
  QApplication app(argc, argv);
  SysInfo::instance().init();
  MainWindow w;
  w.show();
  {

    std::chrono::duration<double> timestamp =
        std::chrono::high_resolution_clock::now() - g_start_time;
    (std::cout) << (std::setw(10)) << (timestamp.count()) << (" ")
                << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
                << (__LINE__) << (" ") << (__func__) << (" ")
                << ("window shown, start event loop") << (" ") << (std::endl)
                << (std::flush);
  }
  return app.exec();
}