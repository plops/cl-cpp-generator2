// no preamble

// implementation

#include "PPOCRDet.h"
PPOCRDet::PPOCRDet(std::string model_path, cv::Size input_size,
                   float binary_threshold, float polygon_threshold,
                   int max_candidates, double unclip_ratio,
                   dnn::Backend backend_id, dnn::Target target_id)
    : model_path{model_path}, input_size{input_size},
      binary_threshold{binary_threshold}, polygon_threshold{polygon_threshold},
      max_candidates{max_candidates}, unclip_ratio{unclip_ratio},
      backend_id{backend_id}, target_id{target_id}, model{0} {
  model = TextDetectionModel_DB(readNet(model_path));
  model.setPreferableBackend(backend_id);
  model.setPreferableTarget(target_id);
  model.setBinaryThreshold(binary_threshold);
  model.setPolygonThreshold(polygon_threshold);
  model.setUnclipRatio(unclip_ratio);
  model.setMaxCandidates(max_candidates);
  model.setInputParams(1.0F / 255.F, input_size,
                       cv::Scalar(122.6789F, 116.66877F, 104.0070F));
}
const std::string &PPOCRDet::GetModelPath() const { return model_path; }
void PPOCRDet::SetModelPath(std::string model_path) {
  this->model_path = model_path;
}
const cv::Size &PPOCRDet::GetInputSize() const { return input_size; }
void PPOCRDet::SetInputSize(cv::Size input_size) {
  this->input_size = input_size;
}
const float &PPOCRDet::GetBinaryThreshold() const { return binary_threshold; }
void PPOCRDet::SetBinaryThreshold(float binary_threshold) {
  this->binary_threshold = binary_threshold;
}
const float &PPOCRDet::GetPolygonThreshold() const { return polygon_threshold; }
void PPOCRDet::SetPolygonThreshold(float polygon_threshold) {
  this->polygon_threshold = polygon_threshold;
}
const int &PPOCRDet::GetMaxCandidates() const { return max_candidates; }
void PPOCRDet::SetMaxCandidates(int max_candidates) {
  this->max_candidates = max_candidates;
}
const double &PPOCRDet::GetUnclipRatio() const { return unclip_ratio; }
void PPOCRDet::SetUnclipRatio(double unclip_ratio) {
  this->unclip_ratio = unclip_ratio;
}
const dnn::Backend &PPOCRDet::GetBackendId() const { return backend_id; }
void PPOCRDet::SetBackendId(dnn::Backend backend_id) {
  this->backend_id = backend_id;
}
const dnn::Target &PPOCRDet::GetTargetId() const { return target_id; }
void PPOCRDet::SetTargetId(dnn::Target target_id) {
  this->target_id = target_id;
}
const dnn::TextDetectionModel_DB &PPOCRDet::GetModel() const { return model; }
void PPOCRDet::SetModel(dnn::TextDetectionModel_DB model) {
  this->model = model;
}