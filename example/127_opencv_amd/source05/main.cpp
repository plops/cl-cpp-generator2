#include <iostream>
#include <opencv2/opencv.hpp>
using namespace cv;

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;
  auto camera = VideoCapture(0);
  auto title = "Webcam";
  if (!camera.isOpened()) {
    std::cout << "Error: Could not open camera." << std::endl;
    return 1;
  }
  namedWindow(title, WINDOW_AUTOSIZE);
  auto frame = Mat();
  while (true) {
    camera >> frame;
    if (frame.empty()) {
      break;
    }
    imshow(title, frame);
    if (0 <= waitKey(1)) {
      break;
    }
  }

  return 0;
}
