#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <thread>
;
class Points {
  float *x, *y;

public:
  Points();
  Points(float *x, float *y);
};
