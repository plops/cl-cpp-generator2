#ifndef PPOCRDET_H
#define PPOCRDET_H

// heaeder
 
#include <string>
#include <opencv2/opencv.hpp> 
class PPOCRDet  {
        public:
         PPOCRDet (std::string model_path = "text_detection_en_ppocrv3_2023may_int8.onnx", cv::Size input_size = cv::Size(736, 736), float binary_threshold = 0.30F, float polygon_threshold = 0.50F, int max_candidates = 200, double unclip_ratio = 2.0F, cv::dnn::Backend backend_id = cv::dnn::DNN_BACKEND_DEFAULT, cv::dnn::Target target_id = cv::dnn::DNN_TARGET_CPU)       ;   
        std::pair<std::vector<std::vector<cv::Point>>,std::vector<float>> infer (cv::Mat image)       ;   
        const std::string& GetModelPath () const      ;   
        void SetModelPath (std::string model_path)       ;   
        const cv::Size& GetInputSize () const      ;   
        void SetInputSize (cv::Size input_size)       ;   
        const float& GetBinaryThreshold () const      ;   
        void SetBinaryThreshold (float binary_threshold)       ;   
        const float& GetPolygonThreshold () const      ;   
        void SetPolygonThreshold (float polygon_threshold)       ;   
        const int& GetMaxCandidates () const      ;   
        void SetMaxCandidates (int max_candidates)       ;   
        const double& GetUnclipRatio () const      ;   
        void SetUnclipRatio (double unclip_ratio)       ;   
        const cv::dnn::Backend& GetBackendId () const      ;   
        void SetBackendId (cv::dnn::Backend backend_id)       ;   
        const cv::dnn::Target& GetTargetId () const      ;   
        void SetTargetId (cv::dnn::Target target_id)       ;   
        const cv::dnn::TextDetectionModel_DB& GetModel () const      ;   
        void SetModel (cv::dnn::TextDetectionModel_DB model)       ;   
        public:
        std::string model_path;
        cv::Size input_size;
        float binary_threshold;
        float polygon_threshold;
        int max_candidates;
        double unclip_ratio;
        cv::dnn::Backend backend_id;
        cv::dnn::Target target_id;
        cv::dnn::TextDetectionModel_DB model;
};

#endif /* !PPOCRDET_H */