// no preamble

#include <chrono>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <thread>
extern std::mutex g_stdout_mutex;
extern std::chrono::time_point<std::chrono::high_resolution_clock> g_start_time;
#include "BoardProcessor.h"
#include <chrono>
#include <thread>
BoardProcessor::BoardProcessor(
    int id_, std::shared_ptr<MessageQueue<ProcessFrameEvent>> events_,
    std::shared_ptr<MessageQueue<ProcessedFrameMessage>> msgs_,
    Charuco charuco_)
    : run{true}, id{id_}, events{events_}, msgs{msgs_}, charuco{charuco_} {}
bool BoardProcessor::get_run() { return run; }
int BoardProcessor::get_id() { return id; }
std::shared_ptr<MessageQueue<ProcessFrameEvent>> BoardProcessor::get_events() {
  return events;
}
std::shared_ptr<MessageQueue<ProcessedFrameMessage>>
BoardProcessor::get_msgs() {
  return msgs;
}
Charuco BoardProcessor::get_charuco() { return charuco; }
void BoardProcessor::process() {
  //

  while (run) {
    using namespace std::chrono_literals;
    std::this_thread::sleep_for(1ms);
    auto event{events->receive()};
    processEvent(event);
  }
  // stopping BoardProcessor
}
void BoardProcessor::processEvent(ProcessFrameEvent event) {
  auto frame{event.get_frame()};
  // detect charuco board

  std::vector<int> markerIds;
  std::vector<std::vector<cv::Point2f>> markerCorners;
  cv::aruco::detectMarkers(frame, charuco.board->dictionary, markerCorners,
                           markerIds, charuco.params);
  if ((0) < (markerIds.size())) {
    std::vector<cv::Point2f> charucoCorners;
    std::vector<int> charucoIds;
    cv::aruco::interpolateCornersCharuco(
        markerCorners, markerIds, frame, charuco.board, charucoCorners,
        charucoIds, charuco.camera_matrix, charuco.dist_coeffs);
    if ((0) < (charucoIds.size())) {
      auto color{cv::Scalar(255, 0, 255)};
      cv::aruco::drawDetectedCornersCharuco(frame, charucoCorners, charucoIds,
                                            color);
    }
  }
  auto msg{ProcessedFrameMessage(
      event.get_frame_idx(), event.get_seconds(),
      event.get_time_point_00_capture(), event.get_time_point_01_conversion(),
      std::chrono::high_resolution_clock::now(), frame)};
  auto sentCondition{std::async(std::launch::async,
                                &MessageQueue<ProcessedFrameMessage>::send,
                                msgs, std::move(msg))};
}
void BoardProcessor::stop() { (run) = (false); }