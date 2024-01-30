// no preamble

#include "CpuAffinityManagerBase.h"
#include <sched.h>
#include <stdexcept>
CpuAffinityManagerBase::CpuAffinityManagerBase(pid_t pid) : pid_(pid) {
  auto cpuset{cpu_set_t()};
  if (0 == sched_getaffinity(pid_, sizeof(cpu_set_t), &cpuset)) {
    for (auto i = 0; i < 12; i += 1) {
      if (CPU_ISSET(i, &cpuset)) {
        selected_cpus_.set(i);
      }
    }
  } else {
    throw std::runtime_error("Failed to get CPU affinity");
  }
}
std::bitset<12> CpuAffinityManagerBase::GetSelectedCpus() {
  return selected_cpus_;
}
void CpuAffinityManagerBase::SetSelectedCpus(std::bitset<12> selected_cpus) {
  selected_cpus_ = selected_cpus;
}
std::bitset<12> CpuAffinityManagerBase::GetAffinity() {
  auto cpuset{cpu_set_t()};
  if (0 == sched_getaffinity(pid_, sizeof(cpu_set_t), &cpuset)) {
    auto affinity{std::bitset<12>()};
    for (auto i = 0; i < 12; i += 1) {
      if (CPU_ISSET(i, &cpuset)) {
        affinity.set(i);
      }
    }
    return affinity;
  }
  throw std::runtime_error("Failed to get CPU affinity");
}
void CpuAffinityManagerBase::ApplyAffinity() {
  auto cpuset{cpu_set_t()};
  CPU_ZERO(&cpuset);
  for (auto i = 0; i < 12; i += 1) {
    if (selected_cpus_[i]) {
      CPU_SET(i, &cpuset);
    }
  }
  if (0 != sched_setaffinity(pid_, sizeof(cpu_set_t), &cpuset)) {
    throw std::runtime_error("Failed to set CPU affinity");
  }
}