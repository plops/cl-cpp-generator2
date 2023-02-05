#pragma once
#include "FancyWindow.hpp"
#include <concepts>
#include <cstring>
#include <type_traits>

int main(int argc, char **argv) {
  auto w = FancyWindow({.Width = 320, .Height = 240});

  return 0;
}
