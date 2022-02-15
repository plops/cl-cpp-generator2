// no preamble
;
#include <chrono>
#include <iomanip>
#include <iostream>
#include <thread>
extern std::chrono::time_point<std::chrono::high_resolution_clock> g_start_time;
#include "Charuco.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_opengl3_loader.h"
#include <GLFW/glfw3.h>
Charuco::Charuco(int squares_x_, int squares_y_, float square_length_,
                 float marker_length_, std::string cap_fn_)
    : squares_x(squares_x_), squares_y(squares_y_),
      square_length(square_length_), marker_length(marker_length_),
      dict_int(cv::aruco::DICT_6X6_250),
      board_dict(cv::aruco::getPredefinedDictionary(dict_int)),
      board(cv::aruco::CharucoBoard::create(squares_x, squares_y, square_length,
                                            marker_length, board_dict)),
      params(cv::aruco::DetectorParameters::create()), textures({0, 0}),
      cap_fn(cap_fn_), cap(cv::VideoCapture(cap_fn)) {
  // opencv initialization
  ;
  board->draw(cv::Size(1600, 800), board_img3, 10, 1);
  cv::cvtColor(board_img3, board_img, cv::COLOR_BGR2RGBA);
  if (cap.isOpened()) {
    {

      std::chrono::duration<double> timestamp =
          std::chrono::high_resolution_clock::now() - g_start_time;
      (std::cout) << (std::setw(10)) << (timestamp.count()) << (" ")
                  << (std::this_thread::get_id()) << (" ") << (__FILE__)
                  << (":") << (__LINE__) << (" ") << (__func__) << (" ")
                  << ("opened video device") << (" ") << (std::setw(8))
                  << (" cap_fn='") << (cap_fn) << ("'") << (std::setw(8))
                  << (" cap.getBackendName()='") << (cap.getBackendName())
                  << ("'") << (std::endl) << (std::flush);
    }
  } else {
    {

      std::chrono::duration<double> timestamp =
          std::chrono::high_resolution_clock::now() - g_start_time;
      (std::cout) << (std::setw(10)) << (timestamp.count()) << (" ")
                  << (std::this_thread::get_id()) << (" ") << (__FILE__)
                  << (":") << (__LINE__) << (" ") << (__func__) << (" ")
                  << ("failed to open video device") << (" ") << (std::setw(8))
                  << (" cap_fn='") << (cap_fn) << ("'") << (std::endl)
                  << (std::flush);
    }
  }
}
void Charuco::Init() {
  // this function requires Capture() to have been called at least once, so that
  // we know the image data.
  ;
  glGenTextures(2, textures.data());
  {
    auto texture = textures[0];
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img.cols, img.rows, 0, GL_RGBA,
                 GL_UNSIGNED_BYTE, img.data);
  }
  {
    auto texture = textures[1];
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
  glBindTexture(GL_TEXTURE_2D, textures[0]);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img.cols, img.rows, 0, GL_RGBA,
               GL_UNSIGNED_BYTE, img.data);
  ImGui::Image(reinterpret_cast<void *>(textures[0]),
               ImVec2(img.cols, img.rows));
  ImGui::End();
  ImGui::Begin("board_img.data");
  glBindTexture(GL_TEXTURE_2D, textures[1]);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, board_img.cols, board_img.rows, 0,
               GL_RGBA, GL_UNSIGNED_BYTE, board_img.data);
  ImGui::Image(reinterpret_cast<void *>(textures[1]),
               ImVec2(board_img.cols, board_img.rows));
  ImGui::End();
}
cv::Mat Charuco::Capture() {
  (cap) >> (img3);
  cv::cvtColor(img3, img, cv::COLOR_BGR2RGBA);
  return img;
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
cv::Mat Charuco::get_camera_matrix() { return camera_matrix; }
cv::Mat Charuco::get_dist_coeffs() { return dist_coeffs; }
std::string Charuco::get_cap_fn() { return cap_fn; }
cv::VideoCapture Charuco::get_cap() { return cap; }
void Charuco::Shutdown() {
  {

    std::chrono::duration<double> timestamp =
        std::chrono::high_resolution_clock::now() - g_start_time;
    (std::cout) << (std::setw(10)) << (timestamp.count()) << (" ")
                << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
                << (__LINE__) << (" ") << (__func__) << (" ")
                << ("delete textures") << (" ") << (std::endl) << (std::flush);
  }
  glDeleteTextures(2, textures.data());
}