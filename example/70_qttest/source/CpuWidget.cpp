#include "SysInfo.h"
#include "SysInfoWidget.h"
#include <QtCharts/QPieSeries>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <thread>
extern std::chrono::time_point<std::chrono::high_resolution_clock> g_start_time;
#include "CpuWidget.h"
#include <QTimer>
#include <QVBoxLayout>
#include <QWidget>
CpuWidget::CpuWidget(QWidget *parent)
    : SysInfoWidget(parent), series_(new QtCharts::QPieSeries(this)) {
  series_->setHoleSize((0.350f));
  series_->append("CPU Load", (30.f));
  series_->append("CPU Free", (70.f));
  auto chart = chartView().chart();
  chart->addSeries(series_);
  chart->setTitle("CPU average load");
}
void CpuWidget::updateSeries() {
  {

    std::chrono::duration<double> timestamp =
        std::chrono::high_resolution_clock::now() - g_start_time;
    (std::cout) << (std::setw(10)) << (timestamp.count()) << (" ")
                << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
                << (__LINE__) << (" ") << (__func__) << (" ") << ("") << (" ")
                << (std::endl) << (std::flush);
  }
  auto cpuLoadAverage = SysInfo::instance().cpuLoadAverage();
  series_->clear();
  series_->append("Load", cpuLoadAverage);
  series_->append("Free", (((1.00e+2f)) - (cpuLoadAverage)));
}