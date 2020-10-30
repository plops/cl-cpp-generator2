
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
const int BITMAPS_COUNT = 106;
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
CharData findCharData(uint8_t *img, int w, int h, int x0, int y0) {
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
  auto iter = color_per_count.rbegin();
  auto count2 = iter->first;
  auto max_count_color1 = iter->second;
  auto max_count_color2 = max_count_color1;
  if (!(((++iter)) == (color_per_count.rend()))) {
    (count2) += (iter->first);
    max_count_color2 = iter->second;
  }
  auto bits = 0;
  auto direct = (((((8) * (4))) / (2))) < (count2);
  if (direct) {
    for (auto y = 0; (y) < (8); (y) += (1)) {
      for (auto x = 0; (x) < (4); (x) += (1)) {
        bits = (bits) << (1);
        auto d1 = 0;
        auto d2 = 0;
        for (auto i = 0; (i) < (3); (i) += (1)) {
          auto shift = ((16) - (((8) * (i))));
          auto c1 = (((max_count_color1) >> (shift)) & (255));
          auto c2 = (((max_count_color2) >> (shift)) & (255));
          auto c =
              img[((i) + (((3) * (((x0) + (x) + (((w) * (((y0) + (y))))))))))];
          (d1) += (((((c1) - (c))) * (((c1) - (c)))));
          (d2) += (((((c2) - (c))) * (((c2) - (c)))));
        }
        if ((d2) < (d1)) {
          bits = ((bits) | (1));
        }
      }
    }
  } else {
    // determine channel with greatest range
    ;
    auto splitIndex = 0;
    auto bestSplit = 0;
    for (auto i = 0; (i) < (3); (i) += (1)) {
      auto delta = ((max[i]) - (min[i]));
      if ((bestSplit) < (delta)) {
        bestSplit = delta;
        splitIndex = i;
      }
    }
    // split at middle instead of median
    ;
    auto splitValue = ((min[splitIndex]) + (((bestSplit) / (2))));
    // bitmap using split and sum the color for both buckets
    ;
    for (auto y = 0; (y) < (8); (y) += (1)) {
      for (auto x = 0; (x) < (4); (x) += (1)) {
        bits = (bits) << (1);
        if ((splitValue) <
            (img[((splitIndex) +
                  (((3) * (((x0) + (x) + (((w) * (((y0) + (y))))))))))])) {
          bits = ((1) | (bits));
        }
      }
    }
  }
  // find the best bitmap match by counting bits that don't match, including the
  // inverted bitmaps
  ;
  auto best_diff = int(8);
  auto best_pattern = static_cast<unsigned int>(0xFFFF);
  auto codepoint = 0x2584;
  auto inverted = false;
  for (auto ii = 0; (ii) < (((BITMAPS_COUNT) / (2))); (ii) += (1)) {
    auto i = ((2) * (ii));
    auto pattern = BITMAPS[i];
    for (auto j = 0; (j) < (2); (j) += (1)) {
      auto diff = int(std::bitset<32>(((pattern) ^ (bits))).count());
      if ((diff) < (best_diff)) {
        // pattern might be inverted
        ;
        best_pattern = BITMAPS[i];
        codepoint = BITMAPS[((i) + (1))];
        best_diff = diff;
        inverted = (best_pattern) != (pattern);
      }
      pattern = ~pattern;
    }
  }
  if (direct) {
    auto result = CharData(0);
    if (inverted) {
      auto tmp = max_count_color1;
      max_count_color1 = max_count_color2;
      max_count_color2 = tmp;
    }
    for (auto i = 0; (i) < (3); (i) += (1)) {
      auto shift = ((16) - (((8) * (i))));
      result.fgColor[i] = (((max_count_color2) >> (shift)) & (255));
      result.bgColor[i] = (((max_count_color1) >> (shift)) & (255));
    }
    result.codePoint = codepoint;
    return result;
  }
  return createCharData(img, w, h, x0, y0, codepoint, best_pattern);
}