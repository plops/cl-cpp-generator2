// no preamble

#include "DiagramBase.h"
#include <format>
#include <stdexcept>
DiagramBase::DiagramBase(unsigned long max_cores, unsigned int max_points,
                         std::string name_y)
    : max_cores_{max_cores}, max_points_{max_points}, diagrams_{},
      name_y_{name_y}, time_points_{} {
  diagrams_.reserve(max_cores_);
  for (auto i = 0; i < max_cores_; i += 1) {
    diagrams_.push_back({std::format("Core {}", i), {}});
  }
}
const unsigned long &DiagramBase::GetMaxCores() const { return max_cores_; }
const unsigned int &DiagramBase::GetMaxPoints() const { return max_points_; }
const std::vector<DiagramData> &DiagramBase::GetDiagrams() const {
  return diagrams_;
}
const std::string &DiagramBase::GetNameY() const { return name_y_; }
const std::deque<float> &DiagramBase::GetTimePoints() const {
  return time_points_;
}