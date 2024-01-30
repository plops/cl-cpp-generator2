#ifndef CPUAFFINITYMANAGERWITHGUI_H
#define CPUAFFINITYMANAGERWITHGUI_H

#include <sched.h>
#include <unistd.h>
#include <bitset>
#include <cstring>
#include <string> 
#include "CpuAffinityManagerBase.h" 
class CpuAffinityManagerWithGui : public CpuAffinityManagerBase {
        public:
        using CpuAffinityManagerBase::CpuAffinityManagerBase; 
        void RenderGui ()       ;   
};

#endif /* !CPUAFFINITYMANAGERWITHGUI_H */