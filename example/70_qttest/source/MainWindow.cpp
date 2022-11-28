// no preamble
;
#include <chrono>
#include <iomanip>
#include <iostream>
#include <qcustomplot.h>
#include <thread>
extern std::chrono::time_point<std::chrono::high_resolution_clock> g_start_time;
#include "MainWindow.h"
#include <QHBoxLayout>
#include <QWidget>
MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), centralWidget_(new QWidget()),
      plot_(new QCustomPlot()), graph_count_(0), cpuWidget_(this),
      memoryWidget_(this) {
  setCentralWidget(centralWidget_);
  auto l = new QHBoxLayout();
  centralWidget_->setLayout(l);
  l->addWidget(&cpuWidget_);
  l->addWidget(&memoryWidget_);
  l->addWidget(plot_);
  setGeometry(400, 250, 542, 390);
  {

    std::chrono::duration<double> timestamp =
        std::chrono::high_resolution_clock::now() - g_start_time;
    (std::cout) << (std::setw(10)) << (timestamp.count()) << (" ")
                << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
                << (__LINE__) << (" ") << (__func__) << (" ") << ("") << (" ")
                << (std::endl) << (std::flush);
  }
}
MainWindow::~MainWindow() {}