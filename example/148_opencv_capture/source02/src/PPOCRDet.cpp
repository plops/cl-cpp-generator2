// no preamble

// implementation

#include "PPOCRDet.h"
PPOCRDet::PPOCRDet(std::string model_path, cv::Size input_size,
                   float binary_threshold, float polygon_threshold,
                   int max_candidates, double unclip_ratio,
                   cv::dnn::Backend backend_id, cv::dnn::Target target_id)
    : model_path{model_path}, input_size{input_size},
      binary_threshold{binary_threshold}, polygon_threshold{polygon_threshold},
      max_candidates{max_candidates}, unclip_ratio{unclip_ratio},
      backend_id{backend_id}, target_id{target_id},
      model{cv::dnn::TextDetectionModel_DB(cv::dnn::readNet(model_path))} {
  model.setPreferableBackend(backend_id);
  model.setPreferableTarget(target_id);
  model.setBinaryThreshold(binary_threshold);
  model.setPolygonThreshold(polygon_threshold);
  model.setUnclipRatio(unclip_ratio);
  model.setMaxCandidates(max_candidates);
  model.setInputParams(1.0F / 255.F, input_size,
                       cv::Scalar(122.6789F, 116.66877F, 104.0070F));
}
std::pair<std::vector<std::vector<cv::Point>>, std::vector<float>>
PPOCRDet::infer(cv::Mat image) {
  CV_Assert(image.rows == input_size.height &&
            "height of input image != net input size");
  CV_Assert(image.cols == input_size.width &&
            "width of input image != net input size");
  auto pt{std::vector<std::vector<cv::Point>>()};
  auto confidence{std::vector<float>()};
  model.detect(image, pt, confidence);
  return std::make_pair<std::vector<std::vector<cv::Point>> &,
                        std::vector<float> &>(pt, confidence);
}
const std::string &PPOCRDet::GetModelPath() const { return model_path; }
void PPOCRDet::SetModelPath(std::string model_path) {
  this->model_path = model_path;
}
const cv::Size &PPOCRDet::GetInputSize() const { return input_size; }
void PPOCRDet::SetInputSize(cv::Size input_size) {
  this->input_size = input_size;
  model.setInputParams(1.0F / 255.F, input_size,
                       cv::Scalar(122.6789F, 116.66877F, 104.0070F));
}
const float &PPOCRDet::GetBinaryThreshold() const { return binary_threshold; }
void PPOCRDet::SetBinaryThreshold(float binary_threshold) {
  this->binary_threshold = binary_threshold;
  model.setBinaryThreshold(binary_threshold);
}
const float &PPOCRDet::GetPolygonThreshold() const { return polygon_threshold; }
void PPOCRDet::SetPolygonThreshold(float polygon_threshold) {
  this->polygon_threshold = polygon_threshold;
  model.setPolygonThreshold(polygon_threshold);
}
const int &PPOCRDet::GetMaxCandidates() const { return max_candidates; }
void PPOCRDet::SetMaxCandidates(int max_candidates) {
  this->max_candidates = max_candidates;
  model.setMaxCandidates(max_candidates);
}
const double &PPOCRDet::GetUnclipRatio() const { return unclip_ratio; }
void PPOCRDet::SetUnclipRatio(double unclip_ratio) {
  this->unclip_ratio = unclip_ratio;
  model.setUnclipRatio(unclip_ratio);
}
const cv::dnn::Backend &PPOCRDet::GetBackendId() const { return backend_id; }
void PPOCRDet::SetBackendId(cv::dnn::Backend backend_id) {
  this->backend_id = backend_id;
  model.setPreferableBackend(backend_id);
}
const cv::dnn::Target &PPOCRDet::GetTargetId() const { return target_id; }
void PPOCRDet::SetTargetId(cv::dnn::Target target_id) {
  this->target_id = target_id;
  model.setPreferableTarget(target_id);
}
const cv::dnn::TextDetectionModel_DB &PPOCRDet::GetModel() const {

  return model;
}
void PPOCRDet::SetModel(cv::dnn::TextDetectionModel_DB model) {
  this->model = model;
}