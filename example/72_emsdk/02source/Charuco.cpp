// no preamble
;
#include <chrono>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <thread>
extern std::mutex g_stdout_mutex;
extern std::chrono::time_point<std::chrono::high_resolution_clock> g_start_time;
#include "Charuco.h"
#include <opencv2/aruco.hpp>
#include <opencv2/aruco/charuco.hpp>
#include <opencv2/aruco/dictionary.hpp>
#include <opencv2/core/cvstd.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
Charuco::Charuco(int squares_x_, int squares_y_, float square_length_,
                 float marker_length_)
    : squares_x(squares_x_), squares_y(squares_y_),
      square_length(square_length_), marker_length(marker_length_),
      dict_int(cv::aruco::DICT_6X6_250),
      board_dict(cv::aruco::getPredefinedDictionary(dict_int)),
      board(cv::aruco::CharucoBoard::create(squares_x, squares_y, square_length,
                                            marker_length, board_dict)),
      params(cv::aruco::DetectorParameters::create()) {
  {

    auto lock = std::unique_lock<std::mutex>(g_stdout_mutex);
    std::chrono::duration<double> timestamp =
        std::chrono::high_resolution_clock::now() - g_start_time;
    (std::cout) << (std::setw(10)) << (timestamp.count()) << (" ")
                << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
                << (__LINE__) << (" ") << (__func__) << (" ")
                << ("opencv initialization") << (" ") << (std::endl)
                << (std::flush);
  }
  board->draw(cv::Size(1600, 800), board_img3, 10, 1);
  cv::cvtColor(board_img3, board_img, cv::COLOR_BGR2RGBA);
  {

    auto lock = std::unique_lock<std::mutex>(g_stdout_mutex);
    std::chrono::duration<double> timestamp =
        std::chrono::high_resolution_clock::now() - g_start_time;
    (std::cout) << (std::setw(10)) << (timestamp.count()) << (" ")
                << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
                << (__LINE__) << (" ") << (__func__) << (" ")
                << ("charuco board has been converted to RGBA") << (" ")
                << (std::endl) << (std::flush);
  }
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
cv::Mat Charuco::get_camera_matrix() { return camera_matrix; }
cv::Mat Charuco::get_dist_coeffs() { return dist_coeffs; }
void Charuco::Shutdown() {}