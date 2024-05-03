#include "Operator.h"
#include "Symbol.h"
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
  // only single digit numbers supported for now

  auto sExpression{std::string("1+2*4-3")};
  auto stkHolding{std::deque<Symbol>()};
  auto stkOutput{std::deque<Symbol>()};
  for (const auto &c : sExpression) {
    if (std::isdigit(c)) {
      // literal straight to output. they are already in order

      stkOutput.push_back({std::string(1, c), Type::Literal_Numeric});
    } else if (mapOps.contains(c)) {
      // symbol is operator

      auto &new_op{mapOps[c]};
      while (!stkHolding.empty()) {
        // ensure holding stack front is an operator

        auto front{stkHolding.front()};
        if (Type::Operator == front.type) {
          if (new_op.precedence <= front.op.precedence) {
            stkOutput.push_back(front);
            stkHolding.pop_front();
          } else {
            break;
          }
        }
      }
      // push new operator on holding stack

      stkHolding.push_front({std::string(1, c), Type::Operator, new_op});
    } else {
      std::cout << std::format("error c='{}'\n", c);
      return 0;
    }
  }
  while (!stkHolding.empty()) {
    stkOutput.push_back(stkHolding.front());
    stkHolding.pop_front();
  }
  std::cout << std::format(" sExpression='{}'\n", sExpression);
  for (const auto &s : stkOutput) {
    std::cout << std::format(" s.symbol='{}'\n", s.symbol);
  }
  return 0;
}
