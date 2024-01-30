#ifndef CPUAFFINITYMANAGER_H
#define CPUAFFINITYMANAGER_H

#include <sched.h>
#include <unistd.h>
#include <bitset>
#include <cstring>
#include <string> 
class CpuAffinityManager  {
        public:
        explicit  CpuAffinityManager (pid_t pid)       ;   
        void ApplyAffinity ()       ;   
        void RenderGui ()       ;   
        private:
        std::bitset<12> selected_cpus_;
        pid_t pid_;
};

#endif /* !CPUAFFINITYMANAGER_H */