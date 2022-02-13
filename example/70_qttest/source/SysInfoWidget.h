#pragma once
class QWidget;
class QVBoxLayout;
class QTimer;
#include <QtCharts/QChartView>
#include <QTimer>
class SysInfoWidget : public QWidget {
        Q_OBJECT;
        public:
        explicit  SysInfoWidget (QWidget* parent = 0, int updateSeriesDelayMs = 31)     ;  
        protected:
        QtCharts::QChartView& chartView ()     ;  
        protected slots:
        virtual void updateSeries ()  =0   ;  
        private:
        QTimer refreshTimer_;
        QtCharts::QChartView chartView_;
};
