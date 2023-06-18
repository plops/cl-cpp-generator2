#include "ArucoCheckerboardDisplay.h"
#include "CheckerboardDisplayInterface.hpp"
#include <fruit/fruit.h>
#include <iostream>
#include <opencv2/core/ocl.hpp>
using fruit::Component;

auto getCheckerboardDisplayComponent() {
  return fruit::createComponent()
      .bind<CheckerboardDisplayInterface, ArucoCheckerboardDisplay>();
}

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;
  std::cout << ""
            << " cv::ocl::haveOpenCL()='" << cv::ocl::haveOpenCL() << "' "
            << std::endl;
  return 0;
}
