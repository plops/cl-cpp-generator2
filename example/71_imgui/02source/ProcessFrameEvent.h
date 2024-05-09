#pragma once
#include <opencv2/core/core.hpp> 
class ProcessFrameEvent  {
        public:
        int batch_idx; 
        int frame_idx; 
        int dim; 
        float fps; 
        double seconds; 
        cv::Mat frame; 
         ProcessFrameEvent (int batch_idx_, int frame_idx_, int dim_, float fps_, double seconds_, cv::Mat frame_)       ;   
        int get_batch_idx ()       ;   
        int get_frame_idx ()       ;   
        int get_dim ()       ;   
        float get_fps ()       ;   
        double get_seconds ()       ;   
        cv::Mat get_frame ()       ;   
};
