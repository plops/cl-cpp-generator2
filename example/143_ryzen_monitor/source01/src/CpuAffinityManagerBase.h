#ifndef CPUAFFINITYMANAGERBASE_H
#define CPUAFFINITYMANAGERBASE_H

#include <sched.h>
#include <unistd.h>
#include <bitset>
#include <cstring>
#include <string> 
class CpuAffinityManagerBase  {
        public:
        explicit  CpuAffinityManagerBase (pid_t pid)       ;   
        std::bitset<12> GetSelectedCpus ()       ;   
        void SetSelectedCpus (std::bitset<12> selected_cpus)       ;   
        std::bitset<12> GetAffinity ()       ;   
        void ApplyAffinity ()       ;   
        protected:
        std::bitset<12> selected_cpus_;
        pid_t pid_;
};

#endif /* !CPUAFFINITYMANAGERBASE_H */