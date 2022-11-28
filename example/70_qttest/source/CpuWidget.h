#pragma once
class QWidget;
class QVBoxLayout;
class QTimer;
#include "SysInfoWidget.h"
#include "SysInfo.h"
#include <QtCharts/QPieSeries>
class CpuWidget : public SysInfoWidget {
        Q_OBJECT;
        public:
        explicit  CpuWidget (QWidget* parent = 0)     ;  
        protected slots:
        void updateSeries ()   override  ;  
        private:
        QtCharts::QPieSeries* series_;
};
