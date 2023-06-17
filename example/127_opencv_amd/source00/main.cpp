#include <iostream>
#include <opencv2/core/ocl.hpp>

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;
  std::cout << ""
            << " cv::ocl::haveOpenCL()='" << cv::ocl::haveOpenCL() << "' "
            << std::endl;
  return 0;
}
