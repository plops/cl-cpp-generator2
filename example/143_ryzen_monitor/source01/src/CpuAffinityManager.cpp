// no preamble

#include "CpuAffinityManager.h"
#include "imgui.h"
#include <sched.h>
#include <stdexcept>
CpuAffinityManager::CpuAffinityManager(pid_t pid) : pid_(pid) {
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
std::bitset<12> CpuAffinityManager::GetSelectedCpus() { return selected_cpus_; }
std::bitset<12> CpuAffinityManager::GetAffinity() {
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
void CpuAffinityManager::ApplyAffinity() {
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
void CpuAffinityManager::RenderGui() {
  ImGui::Begin("CPU Affinity");
  ImGui::Text("Select CPUs for process ID: %d", pid_);
  auto affinityChanged{false};
  for (auto i = 0; i < 12; i += 1) {
    auto label{std::string("CPU ") + std::to_string(i)};
    bool isSelected{selected_cpus_[i]};
    if (ImGui::Checkbox(label.c_str(), &isSelected)) {
      selected_cpus_[i] = isSelected;
      affinityChanged = true;
    }
  }
  if (affinityChanged) {
    ApplyAffinity();
  }
  ImGui::End();
}