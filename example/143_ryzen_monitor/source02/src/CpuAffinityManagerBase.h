#ifndef CPUAFFINITYMANAGERBASE_H
#define CPUAFFINITYMANAGERBASE_H

#include <sched.h>
#include <unistd.h>
#include <bitset>
#include <cstring>
#include <string>
#include <vector> 
/**  @brief The CpuAffinityManagerBase class is used to limit execution of the Ryzen Monitor GUI on one (or more) particular cores.
  
  This class allows for the separation of the impact of the GUI rendering on the diagrams of other cores during benchmarks.
  The class is currently hardcoded for 12 threads of the Ryzen 5625U processor.
  
  The CpuAffinityManagerWithGui class is derived from this one and provides a method to render checkboxes.
  This separation is done so that the Affinity Business Logic can be tested in Unit Tests without having to link in OpenGL into the test.
  
  Note: There is a bug where the program will crash if no core is selected.
*/ 
class CpuAffinityManagerBase  {
        public:
        explicit  CpuAffinityManagerBase (pid_t pid, int threads)       ;   
        const std::vector<bool>& GetSelectedCpus () const      ;   
        void SetSelectedCpus (const std::vector<bool>& selected_cpus)       ;   
        std::vector<bool> GetAffinity () const      ;   
        void ApplyAffinity ()       ;   
        std::vector<bool> selected_cpus_;
        pid_t pid_;
        int threads_;
};

#endif /* !CPUAFFINITYMANAGERBASE_H */