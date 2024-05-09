#include "BoardProcessor.h"
#include "Charuco.h"
#include "MainWindow.h"
#include "MessageQueue.h"
#include "ProcessFrameEvent.h"
#include "ProcessedFrameMessage.h"
#include "ScrollingBuffer.h"
#include <GL/glew.h>
#include <chrono>
#include <future>
#include <mutex>
#include <thread>
std::chrono::time_point<std::chrono::high_resolution_clock> g_start_time;
std::mutex g_stdout_mutex;
#include "GraphicsFramework.h"
#include "implot.h"
#include <algorithm>
#include <cxxabi.h>
#include <imgui.h>
#include <memory>
#include <opencv2/core/mat.hpp>
#include <ratio>
#include <system_error>
#include <utility>
#include <vector>

int main(int argc, char **argv) {
  (g_start_time) = (std::chrono::high_resolution_clock::now());
  {
    // start

    auto framework{GraphicsFramework()};
    auto eventQueue{std::make_shared<MessageQueue<ProcessFrameEvent>>()};
    auto msgQueue{std::make_shared<MessageQueue<ProcessedFrameMessage>>()};
    auto charuco{Charuco()};
    auto processor_thread_should_run{true};
    auto board_processor{BoardProcessor(0, eventQueue, msgQueue, charuco)};
    auto processor_thread{
        std::thread(&BoardProcessor::process, board_processor)};
    auto capture_thread_should_run{true};
    auto capture_thread{std::thread([&capture_thread_should_run, &eventQueue,
                                     &charuco]() {
      charuco.Capture();
      charuco.Init();
      // started capture thread

      auto frame_count{0};
      while (capture_thread_should_run) {
        auto time_point_00_capture{std::chrono::high_resolution_clock::now()};
        auto gray{charuco.Capture()};
        auto time_point_01_conversion{
            std::chrono::high_resolution_clock::now()};
        std::chrono::duration<double> _timestamp =
            std::chrono::high_resolution_clock::now() - g_start_time;
        auto process_frame_event{ProcessFrameEvent(
            frame_count, _timestamp.count(), time_point_00_capture,
            time_point_01_conversion, gray)};
        auto sentCondition{std::async(
            std::launch::async, &MessageQueue<ProcessFrameEvent>::send,
            eventQueue, std::move(process_frame_event))};
        (frame_count)++;
      }
      // shutdown capture thread

      charuco.Shutdown();
    })};
    MainWindow M;
    M.Init(framework.getWindow(), "#version 130");
    while (!framework.WindowShouldClose()) {
      framework.PollEvents();
      M.NewFrame();
      M.Update([&msgQueue, &charuco]() {
        {
          static bool board_texture_is_initialized = false;
          static int board_w = 0;
          static int board_h = 0;
          static std::vector<GLuint> textures({0});
          if (board_texture_is_initialized) {
            ImGui::Begin("board");
            glBindTexture(GL_TEXTURE_2D, (textures)[(0)]);
            ImGui::Image(reinterpret_cast<void *>((textures)[(0)]),
                         ImVec2(board_w, board_h));
            ImGui::End();
          }
          auto board_img{charuco.get_board_img()};
          (board_w) = (board_img.cols);
          (board_h) = (board_img.rows);
          if (board_texture_is_initialized) {
            glBindTexture(GL_TEXTURE_2D, (textures)[(0)]);
            glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, board_img.cols,
                         board_img.rows, 0, GL_RGBA, GL_UNSIGNED_BYTE,
                         board_img.data);
          } else {
            glGenTextures(textures.size(), textures.data());
            glBindTexture(GL_TEXTURE_2D, (textures)[(0)]);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, board_img.cols,
                         board_img.rows, 0, GL_RGBA, GL_UNSIGNED_BYTE,
                         board_img.data);
            (board_texture_is_initialized) = (true);
          }
          ImGui::Begin("board");
          glBindTexture(GL_TEXTURE_2D, (textures)[(0)]);
          ImGui::Image(reinterpret_cast<void *>((textures)[(0)]),
                       ImVec2(board_w, board_h));
          ImGui::End();
        }
        {
          static int w = 0;
          static int h = 0;
          static bool texture_is_initialized = false;
          static std::vector<GLuint> textures({0});
          static ImPlotAxisFlags timeplot_flags_x =
              ImPlotAxisFlags_NoTickLabels;
          static ImPlotAxisFlags timeplot_flags_y = ImPlotAxisFlags_AutoFit;
          static float history = 10.0f;
          static float time = 0;
          (time) += (ImGui::GetIO().DeltaTime);
          static ScrollingBuffer data_00;
          static ScrollingBuffer data_01;
          static ScrollingBuffer data_02;
          if (msgQueue->empty()) {
            if (texture_is_initialized) {
              ImGui::Begin("camera");
              glBindTexture(GL_TEXTURE_2D, (textures)[(0)]);
              ImGui::Image(reinterpret_cast<void *>((textures)[(0)]),
                           ImVec2(w, h));
              ImGui::End();
              ImGui::Begin("time durations [ms]");
              {
                // no extra code

                if (ImPlot::BeginPlot("##Scrolling_capture", ImVec2(-1, 150))) {
                  ImPlot::SetupAxes(nullptr, nullptr, timeplot_flags_x,
                                    timeplot_flags_y);
                  ImPlot::SetupAxisLimits(ImAxis_X1, (time) - (history), time,
                                          ImGuiCond_Always);
                  ImPlot::SetupAxisLimits(ImAxis_Y1, 40, 60);
                  ImPlot::PlotLine("capture", &(data_00).(data)[(0)].x,
                                   &(data_00).(data)[(0)].y,
                                   data_00.data.size(), data_00.offset,
                                   (2) * (sizeof(float)));
                  ImPlot::EndPlot();
                }
              }
              {
                // no extra code

                if (ImPlot::BeginPlot("##Scrolling_conversion",
                                      ImVec2(-1, 150))) {
                  ImPlot::SetupAxes(nullptr, nullptr, timeplot_flags_x,
                                    timeplot_flags_y);
                  ImPlot::SetupAxisLimits(ImAxis_X1, (time) - (history), time,
                                          ImGuiCond_Always);
                  ImPlot::SetupAxisLimits(ImAxis_Y1, 0, 3);
                  ImPlot::PlotLine("conversion", &(data_01).(data)[(0)].x,
                                   &(data_01).(data)[(0)].y,
                                   data_01.data.size(), data_01.offset,
                                   (2) * (sizeof(float)));
                  ImPlot::EndPlot();
                }
              }
              {
                // no extra code

                if (ImPlot::BeginPlot("##Scrolling_processed",
                                      ImVec2(-1, 150))) {
                  ImPlot::SetupAxes(nullptr, nullptr, 0, timeplot_flags_y);
                  ImPlot::SetupAxisLimits(ImAxis_X1, (time) - (history), time,
                                          ImGuiCond_Always);
                  ImPlot::SetupAxisLimits(ImAxis_Y1, 0, 60);
                  ImPlot::PlotLine("processed", &(data_02).(data)[(0)].x,
                                   &(data_02).(data)[(0)].y,
                                   data_02.data.size(), data_02.offset,
                                   (2) * (sizeof(float)));
                  ImPlot::EndPlot();
                }
              }
              ImGui::End();
            }
          } else {
            auto msg{msgQueue->receive()};
            std::chrono::duration<double> _timestamp =
                std::chrono::high_resolution_clock::now() - g_start_time;
            auto frame{msg.get_frame()};
            ImGui::Begin("time durations [ms]");
            {
              auto p{msg.get_time_point_00_capture()};
              auto p0{msg.get_time_point_00_capture()};
              auto p1{msg.get_time_point_01_conversion()};
              auto p2{msg.get_time_point_02_processed()};
              // compute time derivative of acquisition time stamps

              static float old_x = time;
              static auto old_y = p;
              std::chrono::duration<float> y{(1000) * ((p) - (old_y))};
              data_00.AddPoint(static_cast<float>(time),
                               static_cast<float>(y.count()));
              (old_x) = (time);
              (old_y) = (p);
              if (ImPlot::BeginPlot("##Scrolling_capture", ImVec2(-1, 150))) {
                ImPlot::SetupAxes(nullptr, nullptr, timeplot_flags_x,
                                  timeplot_flags_y);
                ImPlot::SetupAxisLimits(ImAxis_X1, (time) - (history), time,
                                        ImGuiCond_Always);
                ImPlot::SetupAxisLimits(ImAxis_Y1, 40, 60);
                ImPlot::PlotLine("capture", &(data_00).(data)[(0)].x,
                                 &(data_00).(data)[(0)].y, data_00.data.size(),
                                 data_00.offset, (2) * (sizeof(float)));
                ImPlot::EndPlot();
              }
            }
            {
              auto p{msg.get_time_point_01_conversion()};
              auto p0{msg.get_time_point_00_capture()};
              auto p1{msg.get_time_point_01_conversion()};
              auto p2{msg.get_time_point_02_processed()};
              // compute time difference

              std::chrono::duration<float> y{(1000) * ((p1) - (p0))};
              data_01.AddPoint(static_cast<float>(time),
                               static_cast<float>(y.count()));
              if (ImPlot::BeginPlot("##Scrolling_conversion",
                                    ImVec2(-1, 150))) {
                ImPlot::SetupAxes(nullptr, nullptr, timeplot_flags_x,
                                  timeplot_flags_y);
                ImPlot::SetupAxisLimits(ImAxis_X1, (time) - (history), time,
                                        ImGuiCond_Always);
                ImPlot::SetupAxisLimits(ImAxis_Y1, 0, 3);
                ImPlot::PlotLine("conversion", &(data_01).(data)[(0)].x,
                                 &(data_01).(data)[(0)].y, data_01.data.size(),
                                 data_01.offset, (2) * (sizeof(float)));
                ImPlot::EndPlot();
              }
            }
            {
              auto p{msg.get_time_point_02_processed()};
              auto p0{msg.get_time_point_00_capture()};
              auto p1{msg.get_time_point_01_conversion()};
              auto p2{msg.get_time_point_02_processed()};
              // compute time difference

              std::chrono::duration<float> y{(1000) * ((p2) - (p1))};
              data_02.AddPoint(static_cast<float>(time),
                               static_cast<float>(y.count()));
              if (ImPlot::BeginPlot("##Scrolling_processed", ImVec2(-1, 150))) {
                ImPlot::SetupAxes(nullptr, nullptr, 0, timeplot_flags_y);
                ImPlot::SetupAxisLimits(ImAxis_X1, (time) - (history), time,
                                        ImGuiCond_Always);
                ImPlot::SetupAxisLimits(ImAxis_Y1, 0, 60);
                ImPlot::PlotLine("processed", &(data_02).(data)[(0)].x,
                                 &(data_02).(data)[(0)].y, data_02.data.size(),
                                 data_02.offset, (2) * (sizeof(float)));
                ImPlot::EndPlot();
              }
            }
            ImGui::End();
            (w) = (frame.cols);
            (h) = (frame.rows);
            if (texture_is_initialized) {
              glBindTexture(GL_TEXTURE_2D, (textures)[(0)]);
              glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
              glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, frame.cols, frame.rows, 0,
                           GL_BGR, GL_UNSIGNED_BYTE, frame.data);
            } else {
              glGenTextures(textures.size(), textures.data());
              glBindTexture(GL_TEXTURE_2D, (textures)[(0)]);
              glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
              glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
              glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
              glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, frame.cols, frame.rows, 0,
                           GL_BGR, GL_UNSIGNED_BYTE, frame.data);
              (texture_is_initialized) = (true);
            }
            ImGui::Begin("camera");
            glBindTexture(GL_TEXTURE_2D, (textures)[(0)]);
            ImGui::Image(reinterpret_cast<void *>((textures)[(0)]),
                         ImVec2(w, h));
            ImGui::End();
          }
        }
      });
      M.Render(framework.getWindow());
    }
    // run various cleanup functions

    // wait for capture thread to exit

    (capture_thread_should_run) = (false);
    capture_thread.join();
    M.Shutdown();
    // leave program

    return 0;
  }
}
