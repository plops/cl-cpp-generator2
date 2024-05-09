#include "star_tracker.h"
Points::Points() : x{NULL}, y{NULL} {}
Points::Points(float *x, float *y) : x{x}, y{y} {}
using namespace std;
using namespace cv;

int main(int argc, char **argv) {
  auto fn{0};
  auto cap{VideoCapture(fn)};
  if (!(cap.isOpened())) {
    {

      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ") << ("error opening file") << (" ") << (std::setw(8))
                  << (" fn='") << (fn) << ("'") << (std::endl) << (std::flush);
    }
    return -1;
  }
  while (1) {
    Mat frame;
    (cap) >> (frame);
    if (frame.empty()) {
      break;
    }
    imshow("frame", frame);
    auto c{static_cast<char>(waitKey(25))};
    if ((27) == (c)) {
      break;
    }
  }
  cap.release();
  destroyAllWindows();
  return 0;
}
