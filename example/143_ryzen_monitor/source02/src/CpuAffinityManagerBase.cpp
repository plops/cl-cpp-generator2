// no preamble
#include "CpuAffinityManagerBase.h"
#include <sched.h>
#include <stdexcept>
CpuAffinityManagerBase::CpuAffinityManagerBase(pid_t pid, int threads)
    : selected_cpus_{std::vector<bool>(threads, false)}, pid_{pid},
      threads_{threads} {
  auto cpuset{cpu_set_t()};
  if (0 == sched_getaffinity(pid_, sizeof(cpu_set_t), &cpuset)) {
    for (decltype(0 + threads_ + 1) i = 0; i < threads_; i += 1) {
      if (CPU_ISSET(i, &cpuset)) {
        selected_cpus_[i] = true;
      }
    }
  } else {
    throw std::runtime_error("Failed to get CPU affinity");
  }
}
const std::vector<bool> &CpuAffinityManagerBase::GetSelectedCpus() const {
  return selected_cpus_;
}
void CpuAffinityManagerBase::SetSelectedCpus(
    const std::vector<bool> &selected_cpus) {
  selected_cpus_ = selected_cpus;
}
std::vector<bool> CpuAffinityManagerBase::GetAffinity() const {
  auto cpuset{cpu_set_t()};
  if (0 == sched_getaffinity(pid_, sizeof(cpu_set_t), &cpuset)) {
    auto affinity{std::vector<bool>(threads_, false)};
    for (decltype(0 + threads_ + 1) i = 0; i < threads_; i += 1) {
      if (CPU_ISSET(i, &cpuset)) {
        affinity[i] = true;
      }
    }
    return affinity;
  }
  throw std::runtime_error("Failed to get CPU affinity");
}
void CpuAffinityManagerBase::ApplyAffinity() {
  auto cpuset{cpu_set_t()};
  CPU_ZERO(&cpuset);
  for (decltype(0 + threads_ + 1) i = 0; i < threads_; i += 1) {
    if (selected_cpus_[i]) {
      CPU_SET(i, &cpuset);
    }
  }
  if (0 != sched_setaffinity(pid_, sizeof(cpu_set_t), &cpuset)) {
    throw std::runtime_error("Failed to set CPU affinity");
  }
}