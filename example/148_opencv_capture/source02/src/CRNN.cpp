// no preamble

// implementation

#include "CRNN.h"
#include "charset_32_94_3944.h"
CRNN::CRNN(std::string model_path, cv::dnn::Backend backend_id,
           cv::dnn::Target target_id)
    : model_path{model_path}, input_size{cv::Size(100, 32)},
      backend_id{backend_id}, target_id{target_id},
      model{cv::dnn::readNet(model_path)},
      charset{loadCharset("CHARSET_EN_36")}, target_vertices{([this]() {
        auto target_vertices{cv::Mat(4, 1, CV_32FC2)};
        target_vertices.row(0) = cv::Vec2f(0, this->input_size.height - 1);
        target_vertices.row(1) = cv::Vec2f(0, 0);
        target_vertices.row(2) = cv::Vec2f(this->input_size.width - 1, 0);
        target_vertices.row(3) =
            cv::Vec2f(this->input_size.width - 1, this->input_size.height - 1);
        return target_vertices;
      })()} {}
cv::Mat CRNN::preprocess(cv::Mat image, cv::Mat rbbox) {
  // remove conf, reshape and ensure all is np.float32

  auto vertices{cv::Mat()};
  rbbox.reshape(2, 4).convertTo(vertices, CV_32FC2);
  auto rotationMatrix{cv::getPerspectiveTransform(vertices, target_vertices)};
  auto cropped{cv::Mat()};
  cv::warpPerspective(image, cropped, rotationMatrix, input_size);
  // CN can detect digits 0-9 letters a-zA-z and some special characters

  // FIXME: what about EN?

  cv::cvtColor(cropped, cropped, cv::COLOR_BGR2GRAY);
  return cv::dnn::blobFromImage(cropped, 1.0F / 127.50F, input_size,
                                cv::Scalar::all(127.50F));
}
std::u16string CRNN::infer(cv::Mat image, cv::Mat rbbox) {
  auto inputBlob{preprocess(image, rbbox)};
  model.setInput(inputBlob);
  auto outputBlob{model.forward()};
  return postprocess(outputBlob);
}
std::u16string CRNN::postprocess(cv::Mat outputBlob) {
  auto character{outputBlob.reshape(1, outputBlob.size[0])};
  auto text{std::u16string(u"")};
  for (auto i = 0; i < character.rows; i += 1) {
    auto minVal{1.0e+12};
    auto maxVal{-1.0e+12};
    auto maxIdx{cv::Point()};
    cv::minMaxLoc(character.row(i), &minVal, &maxVal, nullptr, &maxIdx);
    // decode characters from outputBlob

    if (0 == maxIdx.x) {
      text += u"-";
    } else {
      text += charset[(maxIdx.x - 1)];
    }
  }
  // adjacent same letters and background text must be removed to get final
  // output

  auto textFilter{std::u16string(u"")};
  for (auto i = 0; i < text.size(); i += 1) {
    if (u'-' != text[i]) {
      if (!(0 < i && text[i] == text[(i - 1)])) {
        textFilter += text[i];
      }
    }
  }
  return textFilter;
}
const std::string &CRNN::GetModelPath() const { return model_path; }
void CRNN::SetModelPath(std::string model_path) {
  this->model_path = model_path;
}
const cv::Size &CRNN::GetInputSize() const { return input_size; }
void CRNN::SetInputSize(cv::Size input_size) { this->input_size = input_size; }
const cv::dnn::Backend &CRNN::GetBackendId() const { return backend_id; }
void CRNN::SetBackendId(cv::dnn::Backend backend_id) {
  this->backend_id = backend_id;
  model.setPreferableBackend(backend_id);
}
const cv::dnn::Target &CRNN::GetTargetId() const { return target_id; }
void CRNN::SetTargetId(cv::dnn::Target target_id) {
  this->target_id = target_id;
  model.setPreferableTarget(target_id);
}
const cv::dnn::Net &CRNN::GetModel() const { return model; }
void CRNN::SetModel(cv::dnn::Net model) { this->model = model; }
const std::vector<std::u16string> &CRNN::GetCharset() const { return charset; }
void CRNN::SetCharset(std::vector<std::u16string> charset) {
  this->charset = charset;
}
const cv::Mat &CRNN::GetTargetVertices() const { return target_vertices; }
void CRNN::SetTargetVertices(cv::Mat target_vertices) {
  this->target_vertices = target_vertices;
}