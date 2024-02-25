// no preamble

#include "CpuAffinityManagerWithGui.h"
#include "imgui.h"
#include <stdexcept>
void CpuAffinityManagerWithGui::RenderGui() {
  ImGui::Begin("CPU Affinity");
  ImGui::Text("Select CPUs for process ID: %d", pid_);
  auto affinityChanged{false};
  for (auto i = 0; i < threads_; i += 1) {
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