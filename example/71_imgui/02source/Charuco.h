#pragma once
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/aruco/charuco.hpp>
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
        cv::Mat img;
        cv::Mat img3;
        std::vector<uint32_t> textures;
        cv::Mat camera_matrix;
        cv::Mat dist_coeffs;
        std::string cap_fn;
        cv::VideoCapture cap;
         Charuco (int squares_x_ = 8, int squares_y_ = 4, float square_length_ = (4.00e-2f), float marker_length_ = (2.00e-2f), std::string cap_fn_ = "/dev/video2")     ;  
        void Init ()     ;  
        void Render ()     ;  
        cv::Mat Capture ()     ;  
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
        cv::Mat get_img ()     ;  
        cv::Mat get_img3 ()     ;  
        std::vector<uint32_t> get_textures ()     ;  
        cv::Mat get_camera_matrix ()     ;  
        cv::Mat get_dist_coeffs ()     ;  
        std::string get_cap_fn ()     ;  
        cv::VideoCapture get_cap ()     ;  
        void Shutdown ()     ;  
};
