#include "Operator.h"
#include <deque>
#include <format>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

int main(int argc, char **argv) {
  std::cout << std::format("start\n");
  auto mapOps{std::unordered_map<char, Operator>()};
  mapOps['/'] = {4, 2};
  mapOps['*'] = {3, 2};
  mapOps['+'] = {2, 2};
  mapOps['-'] = {1, 2};
  auto sExpression{std::string("1+2*4-3")};
  return 0;
}
