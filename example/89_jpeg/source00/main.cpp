#include "JxlEncode.h"
#include <iostream>
#include <popl.hpp>
#include <spdlog/spdlog.h>

int main(int argc, char **argv) {
  spdlog::info("start  argc='{}'", argc);
  auto op{popl::OptionParser("allowed opitons")};
  auto helpOption{op.add<popl::Switch>("h", "help", "produce help message")};
  auto verboseOption{
      op.add<popl::Switch>("v", "verbose", "produce verbose output")};
  op.parse(argc, argv);
  if (helpOption->count()) {
    (std::cout) << (op) << (std::endl);
    exit(0);
  }
  auto jxl{JxlEncode()};
}
