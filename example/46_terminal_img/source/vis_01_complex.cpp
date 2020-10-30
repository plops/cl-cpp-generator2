
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
const unsigned int BITMAPS[106] = {
    0x0,        0xA0,   0xF,        0x2581, 0xFF,       0x2582,
    0xFFF,      0x2583, 0xFFFF,     0x2584, 0xFFFFF,    0x2585,
    0xFFFFFF,   0x2586, 0xFFFFFFF,  0x2587, 0xEEEEEEEE, 0x258A,
    0xCCCCCCCC, 0x258C, 0x88888888, 0x258E, 0xCCCC,     0x2596,
    0x3333,     0x2597, 0xCCCC0000, 0x2598, 0xCCCC3333, 0x259A,
    0x33330000, 0x259D, 0x3333CCCC, 0x259E, 0x3333FFFF, 0x259F,
    0xFF000,    0x2501, 0x66666666, 0x2503, 0x77666,    0x250F,
    0xEE666,    0x2513, 0x66677000, 0x2517, 0x666EE000, 0x251B,
    0x66677666, 0x2523, 0x666EE666, 0x252B, 0xFF666,    0x2533,
    0x666FF000, 0x253B, 0x666FF666, 0x254B, 0xCC000,    0x2578,
    0x66000,    0x2579, 0x33000,    0x257A, 0x66000,    0x257B,
    0x6600660,  0x254F, 0xF0000,    0x2500, 0xF000,     0x2500,
    0x44444444, 0x2502, 0x22222222, 0x2502, 0xE0000,    0x2574,
    0xE000,     0x2574, 0x44440000, 0x2575, 0x22220000, 0x2575,
    0x30000,    0x2576, 0x3000,     0x2576, 0x4444,     0x2577,
    0x2222,     0x2577, 0x44444444, 0x23A2, 0x22222222, 0x23A5,
    0xF000000,  0x23BA, 0xF00000,   0x23BB, 0xF00,      0x23BC,
    0xF0,       0x23BD, 0x66000,    0x25AA};
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
  if (!((0) == (bg_count))) {
    for (auto i = 0; (i) < (3); (i) += (1)) {
      result.bgColor[i] = ((result.bgColor[i]) / (bg_count));
    }
  }
  if (!((0) == (fg_count))) {
    for (auto i = 0; (i) < (3); (i) += (1)) {
      result.fgColor[i] = ((result.fgColor[i]) / (fg_count));
    }
  }
  return result;
}
CharData findCharData(uint8_t *img, int w, int x0, int y0) {
  int min[3] = {255, 255, 255};
  int max[3] = {0, 0, 0};
  auto count_per_color = std::map<long, int>();
  // max and min value for each color channel
  ;
  for (auto y = 0; (y) < (8); (y) += (1)) {
    for (auto x = 0; (x) < (4); (x) += (1)) {
      auto color = 0;
      for (auto i = 0; (i) < (3); (i) += (1)) {
        auto d = static_cast<int>(
            img[((i) + (((3) * (((x0) + (x) + (((w) * (((y0) + (y))))))))))]);
        min[i] = std::min(min[i], d);
        max[i] = std::max(max[i], d);
        color = (((color) << (8)) | (d));
      }
      (count_per_color[color])++;
    }
  }
  auto color_per_count = std::multimap<int, long>();
  for (const auto &[k, v] : count_per_color) {
    color_per_count.insert(std::pair<int, long>(v, k));
  }
}