#ifndef CRNN_H
#define CRNN_H

// heaeder
 
#include <string>
#include <opencv2/opencv.hpp> 
class CRNN  {
        public:
         CRNN (std::string model_path = "text_recognition_CRNN_EN_2022oct_int8.onnx", cv::dnn::Backend backend_id = cv::dnn::DNN_BACKEND_DEFAULT, cv::dnn::Target target_id = cv::dnn::DNN_TARGET_CPU)       ;   
        cv::Mat preprocess (cv::Mat image, cv::Mat rbbox)       ;   
        std::u16string infer (cv::Mat image, cv::Mat rbbox)       ;   
        std::u16string postprocess (cv::Mat outputBlob)       ;   
        const std::string& GetModelPath () const      ;   
        void SetModelPath (std::string model_path)       ;   
        const cv::Size& GetInputSize () const      ;   
        void SetInputSize (cv::Size input_size)       ;   
        const cv::dnn::Backend& GetBackendId () const      ;   
        void SetBackendId (cv::dnn::Backend backend_id)       ;   
        const cv::dnn::Target& GetTargetId () const      ;   
        void SetTargetId (cv::dnn::Target target_id)       ;   
        const cv::dnn::Net& GetModel () const      ;   
        void SetModel (cv::dnn::Net model)       ;   
        const std::vector<std::u16string>& GetCharset () const      ;   
        void SetCharset (std::vector<std::u16string> charset)       ;   
        const cv::Mat& GetTargetVertices () const      ;   
        void SetTargetVertices (cv::Mat target_vertices)       ;   
        public:
        std::string model_path;
        cv::Size input_size;
        cv::dnn::Backend backend_id;
        cv::dnn::Target target_id;
        cv::dnn::Net model;
        std::vector<std::u16string> charset;
        cv::Mat target_vertices;
};

#endif /* !CRNN_H */