#pragma once
class QWidget;
class QVBoxLayout;
class QTimer;
#include "SysInfoWidget.h"
#include "SysInfo.h"
#include <QtCharts/QLineSeries>
class MemoryWidget : public SysInfoWidget {
        Q_OBJECT;
        public:
        explicit  MemoryWidget (QWidget* parent = 0)     ;  
        protected slots:
        void updateSeries ()   override  ;  
        private:
        QtCharts::QLineSeries* series_;
        qint64 pointPositionX_;
};
