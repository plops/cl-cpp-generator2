// no preamble
;
#include "SysInfoWidget.h"
#include <QTimer>
#include <QVBoxLayout>
#include <QWidget>
#include <QtCharts/QChartView>
SysInfoWidget::SysInfoWidget(QWidget *parent, int updateSeriesDelayMs)
    : QWidget(parent), chartView_(this) {
  refreshTimer_.setInterval(updateSeriesDelayMs);
  connect(&refreshTimer_, &QTimer::timeout, this, &SysInfoWidget::updateSeries);
  refreshTimer_.start(updateSeriesDelayMs);
  chartView_.setRenderHint(QPainter::Antialiasing);
  chartView_.chart()->legend()->setVisible(false);
  auto *layout = new QVBoxLayout(this);
  layout->addWidget(&chartView_);
  setLayout(layout);
}
QtCharts::QChartView &SysInfoWidget::chartView() { return chartView_; }