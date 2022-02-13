#pragma once
#include <QtGlobal>
#include <QVector>
#include "SysInfo.h"
class SysInfoLinuxImpl : public SysInfo {
        public:
        QVector<qulonglong> cpu_load_last_values_;
        QVector<qulonglong> cpuRawData ()     ;  
         SysInfoLinuxImpl ()     ;  
        void init ()   override  ;  
        double cpuLoadAverage ()   override  ;  
        double memoryUsed ()   override  ;  
};
