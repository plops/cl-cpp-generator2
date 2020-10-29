
#include "utils.h"

#include "globals.h"

extern State state;
#include <array>
#include <bitset>
#include <chrono>
#include <cmath>
#include <fcntl.h>
#include <iostream>
#include <map>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>
using namespace std::chrono_literals;

// implementation
#include "vis_00_base.hpp"
CharData createCharData(uint8_t *img, int w, int h, int x0, int y0,
                        int codepoint, int pattern) {
  auto result = CharData(codepoint);
  auto fg_count = 0;
  auto bg_count = 0;
  auto mask = 0x80000000;
  for (auto y = 0; (y) < (8); (y) += (1)) {
    for (auto x = 0; (x) < (4); (x) += (1)) {
      int *avg;
      if (((pattern) & (mask))) {
        avg = result.fgColor.data();
        (fg_count)++;
      } else {
        avg = result.bgColor.data();
        (bg_count)++;
      }
      for (auto i = 0; (i) < (3); (i) += (1)) {
        (avg[i]) +=
            (img[((i) + (((3) * (((x0) + (x) + (((w) * (((y0) + (y))))))))))]);
      }
      mask = (mask) >> (1);
    }
  }
  // average color for each bucket
  ;
  for (auto i = 0; (i) < (3); (i) += (1)) {
    if (!((0) == (bg_count))) {
      result.bgColor[i] = ((result.bgColor[i]) / (bg_count));
    }
    if (!((0) == (fg_count))) {
      result.fgColor[i] = ((result.fgColor[i]) / (fg_count));
    }
  }
  return result;
}