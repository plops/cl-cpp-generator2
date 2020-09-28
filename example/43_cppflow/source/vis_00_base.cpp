
#include "utils.h"

#include "globals.h"

extern State state;
#include <chrono>
#include <cppflow/cppflow.h>
#include <iostream>
#include <thread>

// implementation
int main(int argc, char **argv) {

  (std::cout)
      << (std::setw(10))
      << (std::chrono::high_resolution_clock::now().time_since_epoch().count())
      << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
      << (__LINE__) << (" ") << (__func__) << (" ") << ("start") << (" ")
      << (std::setw(8)) << (" argc='") << (argc) << ("'") << (std::setw(8))
      << (" argv[0]='") << (argv[0]) << ("'") << (std::endl) << (std::flush);
  auto input = cppflow::decode_jpeg(cppflow::read_file(std::string(
      "/home/martin/src/cppflow/examples/efficientnet/my_cat.jpg")));
  input = cppflow::cast(input, TF_UINT8, TF_FLOAT);
  input = cppflow::expand_dims(input, 0);
  auto model = cppflow::model("./model");
  auto output = model(input);

  (std::cout)
      << (std::setw(10))
      << (std::chrono::high_resolution_clock::now().time_since_epoch().count())
      << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
      << (__LINE__) << (" ") << (__func__) << (" ") << ("cat:") << (" ")
      << (std::setw(8)) << (" cppflow::arg_max(output, 1)='")
      << (cppflow::arg_max(output, 1)) << ("'") << (std::endl) << (std::flush);
  return 0;
}