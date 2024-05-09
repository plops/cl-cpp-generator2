#pragma once
#include <opencv2/core/mat.hpp>
#include <chrono>
class ProcessedFrameMessage  {
        public:
        int frame_idx; 
        double seconds; 
        std::chrono::high_resolution_clock::time_point time_point_00_capture; 
        std::chrono::high_resolution_clock::time_point time_point_01_conversion; 
        std::chrono::high_resolution_clock::time_point time_point_02_processed; 
        cv::Mat frame; 
         ProcessedFrameMessage (int frame_idx_, double seconds_, std::chrono::high_resolution_clock::time_point time_point_00_capture_, std::chrono::high_resolution_clock::time_point time_point_01_conversion_, std::chrono::high_resolution_clock::time_point time_point_02_processed_, cv::Mat frame_)       ;   
        int get_frame_idx ()       ;   
        double get_seconds ()       ;   
        std::chrono::high_resolution_clock::time_point get_time_point_00_capture ()       ;   
        std::chrono::high_resolution_clock::time_point get_time_point_01_conversion ()       ;   
        std::chrono::high_resolution_clock::time_point get_time_point_02_processed ()       ;   
        cv::Mat get_frame ()       ;   
};
