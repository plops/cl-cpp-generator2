// no preamble

#include "ProcessFrameEvent.h"
ProcessFrameEvent::ProcessFrameEvent(
    int frame_idx_, double seconds_,
    std::chrono::high_resolution_clock::time_point time_point_00_capture_,
    std::chrono::high_resolution_clock::time_point time_point_01_conversion_,
    cv::Mat frame_)
    : frame_idx{frame_idx_}, seconds{seconds_},
      time_point_00_capture{time_point_00_capture_},
      time_point_01_conversion{time_point_01_conversion_}, frame{frame_} {}
int ProcessFrameEvent::get_frame_idx() { return frame_idx; }
double ProcessFrameEvent::get_seconds() { return seconds; }
std::chrono::high_resolution_clock::time_point
ProcessFrameEvent::get_time_point_00_capture() {
  return time_point_00_capture;
}
std::chrono::high_resolution_clock::time_point
ProcessFrameEvent::get_time_point_01_conversion() {
  return time_point_01_conversion;
}
cv::Mat ProcessFrameEvent::get_frame() { return frame; }