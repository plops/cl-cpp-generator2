#include "MainWindow.h"
#include <cassert>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <opencv2/aruco/charuco.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <thread>
std::chrono::time_point<std::chrono::high_resolution_clock> g_start_time;
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_opengl3_loader.h"
#include "implot.h"
#include <GLFW/glfw3.h>
// https://gist.github.com/TheOpenDevProject/1662fa2bfd8ef087d94ad4ed27746120

class DestroyGLFWwindow {
public:
  void operator()(GLFWwindow *ptr) {
    // Destroy GLFW window context.

    glfwDestroyWindow(ptr);
    glfwTerminate();
  }
};

int main(int argc, char **argv) {
  (g_start_time) = (std::chrono::high_resolution_clock::now());
  // start

  // glfw initialization

  // https://github.com/ocornut/imgui/blob/docking/examples/example_glfw_opengl3/main.cpp

  glfwSetErrorCallback([&](int err, const char *description) {
    // glfw error
  });
  if (!(glfwInit())) {
    // glfwInit failed.
  }
  const char *glsl_version = "#version 130";
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
  std::unique_ptr<GLFWwindow, DestroyGLFWwindow> window;
  auto w{glfwCreateWindow(1800, 1000, "dear imgui example", nullptr, nullptr)};
  if ((nullptr) == (w)) {
    // glfwCreatWindow failed.
  }
  window.reset(w);
  glfwMakeContextCurrent(window.get());
  // enable vsync

  glfwSwapInterval(1);
  // imgui brings its own opengl loader
  // https://github.com/ocornut/imgui/issues/4445

  // opencv initialization

  auto board_dict{cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250)};
  auto board{
      cv::aruco::CharucoBoard::create(8, 4, 4.00e-2F, 2.00e-2F, board_dict)};
  auto params{cv::aruco::DetectorParameters::create()};
  cv::Mat board_img3, board_img;
  board->draw(cv::Size(1600, 800), board_img3, 10, 1);
  cv::cvtColor(board_img3, board_img, cv::COLOR_BGR2RGBA);
  auto cap_fn{"/dev/video2"};
  auto cap{cv::VideoCapture(cap_fn)};
  if (cap.isOpened()) {
    // opened video device

  } else {
    // failed to open video device
  }
  auto cam_w{cap.get(cv::CAP_PROP_FRAME_WIDTH)};
  auto cam_h{cap.get(cv::CAP_PROP_FRAME_HEIGHT)};
  auto cam_fps{cap.get(cv::CAP_PROP_FPS)};
  auto cam_format{cap.get(cv::CAP_PROP_FORMAT)};
  auto cam_brightness{cap.get(cv::CAP_PROP_BRIGHTNESS)};
  auto cam_contrast{cap.get(cv::CAP_PROP_CONTRAST)};
  auto cam_saturation{cap.get(cv::CAP_PROP_SATURATION)};
  auto cam_hue{cap.get(cv::CAP_PROP_HUE)};
  auto cam_gain{cap.get(cv::CAP_PROP_GAIN)};
  auto cam_exposure{cap.get(cv::CAP_PROP_EXPOSURE)};
  auto cam_monochrome{cap.get(cv::CAP_PROP_MONOCHROME)};
  auto cam_sharpness{cap.get(cv::CAP_PROP_SHARPNESS)};
  auto cam_auto_exposure{cap.get(cv::CAP_PROP_AUTO_EXPOSURE)};
  auto cam_gamma{cap.get(cv::CAP_PROP_GAMMA)};
  auto cam_backlight{cap.get(cv::CAP_PROP_BACKLIGHT)};
  auto cam_temperature{cap.get(cv::CAP_PROP_TEMPERATURE)};
  auto cam_auto_wb{cap.get(cv::CAP_PROP_AUTO_WB)};
  auto cam_wb_temperature{cap.get(cv::CAP_PROP_WB_TEMPERATURE)};
  //

  cv::Mat img3, img;
  (cap) >> (img3);
  // received camera image

  cv::cvtColor(img3, img, cv::COLOR_BGR2RGBA);
  // converted camera image

  MainWindow M;
  M.Init(window.get(), glsl_version);
  GLuint textures[2]{{0, 0}};
  glGenTextures(2, textures);
  {
    // prepare texture img.data

    auto texture{(textures)[(0)]};
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img.cols, img.rows, 0, GL_RGBA,
                 GL_UNSIGNED_BYTE, img.data);
  }
  {
    // prepare texture board_img.data

    auto texture{(textures)[(1)]};
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, board_img.cols, board_img.rows, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, board_img.data);
  }
  // uploaded texture

  cv::Mat cameraMatrix, distCoeffs;
  while (!glfwWindowShouldClose(window.get())) {
    (cap) >> (img3);
    // detect charuco board

    std::vector<int> markerIds;
    std::vector<std::vector<cv::Point2f>> markerCorners;
    cv::aruco::detectMarkers(img3, board->dictionary, markerCorners, markerIds,
                             params);
    if ((0) < (markerIds.size())) {
      std::vector<cv::Point2f> charucoCorners;
      std::vector<int> charucoIds;
      cv::aruco::interpolateCornersCharuco(markerCorners, markerIds, img3,
                                           board, charucoCorners, charucoIds,
                                           cameraMatrix, distCoeffs);
      if ((0) < (charucoIds.size())) {
        auto color{cv::Scalar(255, 0, 255)};
        cv::aruco::drawDetectedCornersCharuco(img3, charucoCorners, charucoIds,
                                              color);
      }
    }
    cv::cvtColor(img3, img, cv::COLOR_BGR2RGBA);
    glfwPollEvents();
    M.NewFrame();
    M.Update([&textures, &img, &board_img]() {
      ImGui::Begin("img.data");
      glBindTexture(GL_TEXTURE_2D, (textures)[(0)]);
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img.cols, img.rows, 0, GL_RGBA,
                   GL_UNSIGNED_BYTE, img.data);
      ImGui::Image(reinterpret_cast<void *>((textures)[(0)]),
                   ImVec2(img.cols, img.rows));
      ImGui::End();
      ImGui::Begin("board_img.data");
      glBindTexture(GL_TEXTURE_2D, (textures)[(1)]);
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, board_img.cols, board_img.rows, 0,
                   GL_RGBA, GL_UNSIGNED_BYTE, board_img.data);
      ImGui::Image(reinterpret_cast<void *>((textures)[(1)]),
                   ImVec2(board_img.cols, board_img.rows));
      ImGui::End();
    });
    M.Render(window.get());
  }
  // cleanup

  M.Shutdown();
  // leave program

  return 0;
}
