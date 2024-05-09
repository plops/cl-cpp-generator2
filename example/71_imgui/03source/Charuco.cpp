// no preamble

#include <chrono>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <thread>
extern std::mutex g_stdout_mutex;
extern std::chrono::time_point<std::chrono::high_resolution_clock> g_start_time;
#include "Charuco.h"
#include "imgui_impl_opengl3_loader.h"
#include <imgui.h>
#include <opencv2/aruco.hpp>
#include <opencv2/aruco/charuco.hpp>
#include <opencv2/aruco/dictionary.hpp>
#include <opencv2/core/cvstd.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
Charuco::Charuco(int squares_x_, int squares_y_, float square_length_,
                 float marker_length_, std::string cap_fn_)
    : squares_x{squares_x_}, squares_y{squares_y_},
      square_length{square_length_}, marker_length{marker_length_},
      dict_int{cv::aruco::DICT_6X6_250},
      board_dict{cv::aruco::getPredefinedDictionary(dict_int)},
      board{cv::aruco::CharucoBoard::create(squares_x, squares_y, square_length,
                                            marker_length, board_dict)},
      params{cv::aruco::DetectorParameters::create()}, textures{{0, 0}},
      textures_dirty{{true, true}}, cap_fn{cap_fn_},
      cap{cv::VideoCapture(cap_fn)} {
  // opencv initialization

  board->draw(cv::Size(1600, 800), board_img3, 10, 1);
  cv::cvtColor(board_img3, board_img, cv::COLOR_BGR2RGBA);
  if (cap.isOpened()) {
    // opened video device

  } else {
    // failed to open video device
  }
}
void Charuco::Init() {
  // this function requires Capture() to have been called at least once, so that
  // we know the image data.

  glGenTextures(2, textures.data());
  {
    auto texture{(textures)[(0)]};
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img.cols, img.rows, 0, GL_RGBA,
                 GL_UNSIGNED_BYTE, img.data);
  }
  {
    auto texture{(textures)[(1)]};
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, board_img.cols, board_img.rows, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, board_img.data);
  }
}
void Charuco::Render() {
  ImGui::Begin("img.data");
  glBindTexture(GL_TEXTURE_2D, (textures)[(0)]);
  if ((textures_dirty)[(0)]) {
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img.cols, img.rows, 0, GL_RGBA,
                 GL_UNSIGNED_BYTE, img.data);
    nil;
  }
  ImGui::Image(reinterpret_cast<void *>((textures)[(0)]),
               ImVec2(img.cols, img.rows));
  ImGui::End();
  ImGui::Begin("board_img.data");
  glBindTexture(GL_TEXTURE_2D, (textures)[(1)]);
  if ((textures_dirty)[(1)]) {
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, board_img.cols, board_img.rows, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, board_img.data);
    ((textures_dirty)[(1)]) = (false);
  }
  ImGui::Image(reinterpret_cast<void *>((textures)[(1)]),
               ImVec2(board_img.cols, board_img.rows));
  ImGui::End();
}
cv::Mat Charuco::Capture() {
  (cap) >> (img3);
  return img3;
}
int Charuco::get_squares_x() { return squares_x; }
int Charuco::get_squares_y() { return squares_y; }
float Charuco::get_square_length() { return square_length; }
float Charuco::get_marker_length() { return marker_length; }
int Charuco::get_dict_int() { return dict_int; }
cv::Ptr<cv::aruco::Dictionary> Charuco::get_board_dict() { return board_dict; }
cv::Ptr<cv::aruco::CharucoBoard> Charuco::get_board() { return board; }
cv::Ptr<cv::aruco::DetectorParameters> Charuco::get_params() { return params; }
cv::Mat Charuco::get_board_img() { return board_img; }
cv::Mat Charuco::get_board_img3() { return board_img3; }
cv::Mat Charuco::get_img() { return img; }
cv::Mat Charuco::get_img3() { return img3; }
std::vector<uint32_t> Charuco::get_textures() { return textures; }
std::vector<bool> Charuco::get_textures_dirty() { return textures_dirty; }
cv::Mat Charuco::get_camera_matrix() { return camera_matrix; }
cv::Mat Charuco::get_dist_coeffs() { return dist_coeffs; }
std::string Charuco::get_cap_fn() { return cap_fn; }
cv::VideoCapture Charuco::get_cap() { return cap; }
void Charuco::Shutdown() {
  // delete textures

  glDeleteTextures(2, textures.data());
}