// no preamble

#include "ProcessedFrameMessage.h"
ProcessedFrameMessage::ProcessedFrameMessage(
    int frame_idx_, double seconds_,
    std::chrono::high_resolution_clock::time_point time_point_00_capture_,
    std::chrono::high_resolution_clock::time_point time_point_01_conversion_,
    std::chrono::high_resolution_clock::time_point time_point_02_processed_,
    cv::Mat frame_)
    : frame_idx{frame_idx_}, seconds{seconds_},
      time_point_00_capture{time_point_00_capture_},
      time_point_01_conversion{time_point_01_conversion_},
      time_point_02_processed{time_point_02_processed_}, frame{frame_} {}
int ProcessedFrameMessage::get_frame_idx() { return frame_idx; }
double ProcessedFrameMessage::get_seconds() { return seconds; }
std::chrono::high_resolution_clock::time_point
ProcessedFrameMessage::get_time_point_00_capture() {
  return time_point_00_capture;
}
std::chrono::high_resolution_clock::time_point
ProcessedFrameMessage::get_time_point_01_conversion() {
  return time_point_01_conversion;
}
std::chrono::high_resolution_clock::time_point
ProcessedFrameMessage::get_time_point_02_processed() {
  return time_point_02_processed;
}
cv::Mat ProcessedFrameMessage::get_frame() { return frame; }