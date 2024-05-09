// no preamble

#include <chrono>
#include <iomanip>
#include <iostream>
#include <thread>
extern std::chrono::time_point<std::chrono::high_resolution_clock> g_start_time;
#include "BoardProcessor.h"
#include "ProcessedFrameMessage.h"
#include <chrono>
#include <opencv2/imgproc/imgproc.hpp>
#include <thread>
BoardProcessor::BoardProcessor(
    int id_, std::shared_ptr<MessageQueue<ProcessFrameEvent>> events_,
    std::shared_ptr<MessageQueue<ProcessedFrameMessage>> msgs_)
    : run{true}, id{id_}, events{events_}, msgs{msgs_} {}
bool BoardProcessor::get_run() { return run; }
int BoardProcessor::get_id() { return id; }
std::shared_ptr<MessageQueue<ProcessFrameEvent>> BoardProcessor::get_events() {
  return events;
}
std::shared_ptr<MessageQueue<ProcessedFrameMessage>>
BoardProcessor::get_msgs() {
  return msgs;
}
void BoardProcessor::process() {
  while (run) {
    using namespace std::chrono_literals;
    std::this_thread::sleep_for(1ms);
    auto event{events->receive()};
    processEvent(event);
  }
  // stopping BoardProcessor
}
void BoardProcessor::processEvent(ProcessFrameEvent event) {
  auto dim{event.get_dim()};
  auto frame{event.get_frame()};
  cv::Mat gray;
  cv::cvtColor(frame, gray, cv::COLOR_RGB2GRAY);
  auto msg{ProcessedFrameMessage(event.get_batch_idx(), event.get_frame_idx(),
                                 event.get_seconds())};
  auto sentCondition{std::async(std::launch::async,
                                &MessageQueue<ProcessedFrameMessage>::send,
                                msgs, std::move(msg))};
}
void BoardProcessor::stop() { (run) = (false); }