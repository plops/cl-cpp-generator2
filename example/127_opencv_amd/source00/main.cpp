#include "ArucoCheckerboardDisplay.h"
#include "CheckerboardDisplayInterface.hpp"
#include <fruit/fruit.h>
#include <iostream>
#include <opencv2/core/ocl.hpp>
using fruit::Component;

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;
  std::cout << ""
            << " cv::ocl::haveOpenCL()='" << cv::ocl::haveOpenCL() << "' "
            << std::endl;
  auto injector = fruit::Injector<CheckerBoardDisplayInterface>(
      getCheckerboardDisplayComponent());
  auto display = injector.get<CheckerboardDisplayInterface *>();
  auto dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6x6_250);
  display->displayCheckerboard(5, 7, 100, dictionary);

  return 0;
}
