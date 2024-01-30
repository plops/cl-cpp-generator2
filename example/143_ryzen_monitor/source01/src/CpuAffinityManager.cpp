// no preamble

#include "CpuAffinityManager.h"
#include <sched.h>
CpuAffinityManager::CpuAffinityManager(pid_t pid) : pid_(pid) {
  auto cpuset;
  auto cpu_set_t;
  if (0 == sched_getaffinity(git, sizeof(cpu_set_t), &cpuset)) {
    for (auto i = 0; i < 12; i += 1) {
      if (CPU_ISSET(i, &cpuset)) {
        selectedCpus_.set(i);
      }
    }
  } else {
    throw std::runtime_error("Failed to get CPU affinity");
  }
}
void CpuAffinityManager::ApplyAffinity() {
  auto cpuset{cpu_set_t()};
  CPU_ZERO(&cpuset);
  for (auto i = 0; i < 12; i += 1) {
    if (selectedCpus_[i]) {
      CPU_SET(i, &cpuset);
    }
  }
  if (0 != sched_setaffinity(pid, sizeof(cpu_set_t), &cpuset)) {
    throw std::runtime_error("Failed to set CPU affinity");
  }
}
void CpuAffinityManager::RenderGui() {
  ImGui::Begin("CPU Affinity");
  ImGui::Text("Select CPUs for process ID: %d", pid_);
  auto affinityChanged{false};
  for (auto i = 0; i < 12; i += 1) {
    auto label{std::string("CPU ") + std::to_string(i)};
    if (ImGui::CheckBox(label.c_str(), &selectedCpus_[i])) {
      affinityChanged = true;
    }
  }
  if (affinityChanged) {
    ApplyAffinity();
  }
  ImGui::End();
}