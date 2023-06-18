#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;
  auto img = cv::imread("/home/martin/cow.jpg");
  if (img.empty()) {
    std::cout << "could not open or find the image." << std::endl;
    return -1;
  }
  cv::namedWindow("window", cv::WINDOW_NORMAL);
  cv::imshow("window", img);
  cv::waitKey(0);

  return 0;
}
