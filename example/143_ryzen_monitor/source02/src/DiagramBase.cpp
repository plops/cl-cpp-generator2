// no preamble
#include "DiagramBase.h"
#include <format>
#include <stdexcept>
DiagramBase::DiagramBase(unsigned long max_cores, unsigned int max_points,
                         std::string name_y)
    : max_cores_{max_cores}, max_points_{max_points}, diagrams_{},
      name_y_{name_y}, time_points_{} {
  diagrams_.reserve(max_cores_);
  for (decltype(0 + max_cores_ + 1) i = 0; i < max_cores_; i += 1) {
    diagrams_.push_back({std::format("Core {}", i), {}});
  }
}
void DiagramBase::AddDataPoint(float time, const std::vector<float> &values) {
  if (!(values.size() == diagrams_.size())) {
    throw std::invalid_argument(
        std::format("Number of values doesn't match the number of diagrams. "
                    "expected: {} actual: {}",
                    values.size(), diagrams_.size()));
  }
  if (max_points_ <= time_points_.size()) {
    time_points_.pop_front();
    for (auto &diagram : diagrams_) {
      if (!diagram.values.empty()) {
        diagram.values.pop_front();
      }
    }
  }
  time_points_.push_back(time);
  for (decltype(0 + values.size() + 1) i = 0; i < values.size(); i += 1) {
    diagrams_[i].values.push_back(values[i]);
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