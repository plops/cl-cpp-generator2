#pragma once
#include <opencv2/core/mat.hpp>
#include <chrono> 
class ProcessFrameEvent  {
        public:
        int frame_idx; 
        double seconds; 
        std::chrono::high_resolution_clock::time_point time_point_00_capture; 
        std::chrono::high_resolution_clock::time_point time_point_01_conversion; 
        cv::Mat frame; 
         ProcessFrameEvent (int frame_idx_, double seconds_, std::chrono::high_resolution_clock::time_point time_point_00_capture_, std::chrono::high_resolution_clock::time_point time_point_01_conversion_, cv::Mat frame_)       ;   
        int get_frame_idx ()       ;   
        double get_seconds ()       ;   
        std::chrono::high_resolution_clock::time_point get_time_point_00_capture ()       ;   
        std::chrono::high_resolution_clock::time_point get_time_point_01_conversion ()       ;   
        cv::Mat get_frame ()       ;   
};
