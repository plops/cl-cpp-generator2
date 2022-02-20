#pragma once
#include <opencv2/core/cvstd.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/aruco/charuco.hpp>
namespace cv { namespace aruco { class CharucoBoard;class Dictionary; struct DetectorParameters; }}
class Charuco  {
        public:
        int squares_x;
        int squares_y;
        float square_length;
        float marker_length;
        int dict_int;
        cv::Ptr<cv::aruco::Dictionary> board_dict;
        cv::Ptr<cv::aruco::CharucoBoard>  board;
        cv::Ptr<cv::aruco::DetectorParameters> params;
        cv::Mat board_img;
        cv::Mat board_img3;
        cv::Mat camera_matrix;
        cv::Mat dist_coeffs;
         Charuco (int squares_x_ = 8, int squares_y_ = 4, float square_length_ = (4.00e-2f), float marker_length_ = (2.00e-2f))     ;  
        int get_squares_x ()     ;  
        int get_squares_y ()     ;  
        float get_square_length ()     ;  
        float get_marker_length ()     ;  
        int get_dict_int ()     ;  
        cv::Ptr<cv::aruco::Dictionary> get_board_dict ()     ;  
        cv::Ptr<cv::aruco::CharucoBoard>  get_board ()     ;  
        cv::Ptr<cv::aruco::DetectorParameters> get_params ()     ;  
        cv::Mat get_board_img ()     ;  
        cv::Mat get_board_img3 ()     ;  
        cv::Mat get_camera_matrix ()     ;  
        cv::Mat get_dist_coeffs ()     ;  
        void Shutdown ()     ;  
};
