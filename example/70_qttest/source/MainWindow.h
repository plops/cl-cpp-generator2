#pragma once
class QHBoxLayout;
class QWidget;
#include <vector>
#include <QMainWindow>
#include "CpuWidget.h"
#include "MemoryWidget.h"
class QCustomPlot;
class MainWindow : public QMainWindow {
        Q_OBJECT
        public:
        QWidget* centralWidget_;
        QCustomPlot* plot_;
        int graph_count_;
        CpuWidget cpuWidget_;
        MemoryWidget memoryWidget_;
        explicit  MainWindow (QWidget* parent = 0)     ;  
         ~MainWindow ()     ;  
};
