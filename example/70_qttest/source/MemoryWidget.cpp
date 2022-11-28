#include "SysInfo.h"
#include "SysInfoWidget.h"
#include <QLinearGradient>
#include <QPen>
#include <QtCharts/QAreaSeries>
#include <QtCharts/QLineSeries>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <thread>
extern std::chrono::time_point<std::chrono::high_resolution_clock> g_start_time;
#include "MemoryWidget.h"
#include <QTimer>
#include <QVBoxLayout>
#include <QWidget>
MemoryWidget::MemoryWidget(QWidget *parent)
    : SysInfoWidget(parent), series_(new QtCharts::QLineSeries(this)),
      pointPositionX_(0) {
  auto pen = QPen(0x209FDF);
  auto gradient = QLinearGradient(QPointF(0, 0), QPointF(0, 1));
  auto *areaSeries = new QtCharts::QAreaSeries(series_);
  pen.setWidth(3);
  gradient.setColorAt((1.0), 0x209FDF);
  gradient.setColorAt((0.), 0xBFDFEF);
  gradient.setCoordinateMode(QGradient::ObjectBoundingMode);
  areaSeries->setPen(pen);
  areaSeries->setBrush(gradient);
  auto chart = chartView().chart();
  chart->addSeries(areaSeries);
  chart->setTitle("Memory used");
  chart->createDefaultAxes();
  auto axisX = chart->axes(Qt::Horizontal).back();
  auto axisY = chart->axes(Qt::Vertical).back();
  axisX->setVisible(false);
  axisX->setRange(0, 49);
  axisY->setRange(0, 100);
}
void MemoryWidget::updateSeries() {
  {

    std::chrono::duration<double> timestamp =
        std::chrono::high_resolution_clock::now() - g_start_time;
    (std::cout) << (std::setw(10)) << (timestamp.count()) << (" ")
                << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
                << (__LINE__) << (" ") << (__func__) << (" ") << ("") << (" ")
                << (std::endl) << (std::flush);
  }
  auto memoryUsed = SysInfo::instance().memoryUsed();
  (pointPositionX_)++;
  series_->append(pointPositionX_, memoryUsed);
  if ((50) < (series_->count())) {
    auto chart = chartView().chart();
    chart->scroll(((chart->plotArea().width()) / ((49.))), 0);
    series_->remove(0);
  }
}