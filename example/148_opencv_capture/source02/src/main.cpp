#include "CRNN.h"
#include "PPOCRDet.h"
#include "Screenshot.h"
#include <codecvt>
#include <format>
#include <opencv2/opencv.hpp>

int main(int argc, char **argv) {
  std::cout << std::format("start\n");
  auto img{cv::Mat()};
  auto win{"img"};
  auto frameRate{60.F};
  auto alpha{0.20F};
  auto w{320};
  auto h{256};
  cv::namedWindow(win, cv::WINDOW_AUTOSIZE);
  cv::moveWindow(win, w, 100);
  cv::resizeWindow(win, w, h);
  auto x{20};
  auto y{270};
  auto clipLimit{13};
  auto screen{Screenshot(static_cast<int>(x), static_cast<int>(y), w, h)};
  cv::createTrackbar(
      "x", win, &x, 1920 - w,
      [](int value, void *v) {
        auto screen{reinterpret_cast<Screenshot *>(v)};
        screen->SetX(value);
      },
      reinterpret_cast<void *>(&screen));
  cv::createTrackbar(
      "y", win, &y, 1080 - h,
      [](int value, void *v) {
        auto screen{reinterpret_cast<Screenshot *>(v)};
        screen->SetY(value);
      },
      reinterpret_cast<void *>(&screen));
  cv::createTrackbar("clipLimit", win, &clipLimit, 100);
  try {
    auto binary_threshold{0.30F};
    auto polygon_threshold{0.50F};
    auto max_candidates{200};
    auto unclip_ratio{2.0F};
    auto detector{PPOCRDet(
        "text_detection_en_ppocrv3_2023may_int8.onnx", cv::Size(w, h),
        binary_threshold, polygon_threshold, max_candidates, unclip_ratio,
        cv::dnn::DNN_BACKEND_DEFAULT, cv::dnn::DNN_TARGET_CPU)};
    auto recognizer{CRNN()};
    while (true) {
      screen(img);
      auto img3{cv::Mat()};
      cv::cvtColor(img, img3, cv::COLOR_BGRA2RGB);
      auto result{detector.infer(img3)};
      cv::polylines(img, result.first, true, cv::Scalar(0, 255, 0), 4);
      if (0 < result.first.size() && 0 < result.second.size()) {
        auto texts{std::u16string()};
        auto score{result.second.begin()};
        for (const auto &box : result.first) {
          auto res{cv::Mat(box).reshape(2, 4)};
          texts += u'-' + recognizer.infer(img3, res) + u'-';
        }
        auto converter{
            std::wstring_convert<std::codecvt_utf8<char16_t>, char16_t>()};
        auto ctext{converter.to_bytes(texts)};
        std::cout << std::format(" ctext='{}'\n", ctext);
      }
      if (clipLimit <= 99) {
        auto lab{cv::Mat()};
        cv::cvtColor(img, lab, cv::COLOR_BGR2Lab);
        auto labChannels{std::vector<cv::Mat>()};
        cv::split(lab, labChannels);
        cv::Ptr<cv::CLAHE> clahe{cv::createCLAHE()};
        clahe->setClipLimit(clipLimit);
        auto claheImage{cv::Mat()};
        clahe->apply(labChannels[0], claheImage);
        claheImage.copyTo(labChannels[0]);
        auto processedImage{cv::Mat()};
        cv::merge(labChannels, lab);
        cv::cvtColor(lab, processedImage, cv::COLOR_Lab2BGR);
        cv::imshow(win, processedImage);
      } else {
        cv::imshow(win, img);
      }
      if (27 == cv::waitKey(1000 / 60)) {
        // Exit loop if ESC key is pressed

        break;
      }
    }
  } catch (const std::exception &e) {
    std::cout << std::format(" e.what()='{}'\n", e.what());
    return 1;
  }
  return 0;
}
