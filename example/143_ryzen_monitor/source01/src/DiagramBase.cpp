// no preamble

#include "DiagramBase.h"
#include <format>
#include <stdexcept>
DiagramBase::DiagramBase(int max_cores, int max_points, std::string name_y)
    : max_cores_(max_cores), max_points_(max_points), diagrams_(0),
      name_y_(name_y), time_points_(0) {
  diagrams_.reserve(max_cores_);
  for (auto i = 0; i < max_cores_; i += 1) {
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
  for (auto i = 0; i < values.size(); i += 1) {
    diagrams_[i].values.push_back(values[i]);
  }
}
const int &DiagramBase::GetMaxCores() { return max_cores_; }
const int &DiagramBase::GetMaxPoints() { return max_points_; }
const std::vector<DiagramData> &DiagramBase::GetDiagrams() { return diagrams_; }
const std::string &DiagramBase::GetNameY() { return name_y_; }
const std::deque<float> &DiagramBase::GetTimePoints() { return time_points_; }