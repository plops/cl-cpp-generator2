
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
// bla
uint8_t *img;
const int COLOR_STEPS[6] = {0x0, 0x5F, 0x87, 0xAF, 0xD7, 0xFF};
const int COLOR_STEP_COUNT = 6;
const int GRAYSCALE_STEP_COUNT = 24;
const int GRAYSCALE_STEPS[24] = {
    0x8,  0x12, 0x1C, 0x26, 0x30, 0x3A, 0x44, 0x4E, 0x58, 0x62, 0x6C, 0x76,
    0x80, 0x8A, 0x94, 0x9E, 0xA8, 0xB2, 0xBC, 0xC6, 0xD0, 0xDA, 0xE4, 0xEE};
CharData::CharData(int codepoint) : codePoint(codepoint) {}
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
        (avg[i]) += (img[(
            (i) + (((3) * (((((x0) * (x) * (((w) * (((y0) + (y))))))))))))]);
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
float sqr(float x) { return ((x) * (x)); }
int best_index(int value, const int data[], int count) {
  auto result = 0;
  auto best_diff = std::abs(((data[0]) - (value)));
  for (int i = 1; (i) < (count); (i)++) {
    auto diff = std::abs(((data[i]) - (value)));
    if ((diff) < (best_diff)) {
      result = i;
      best_diff = diff;
    }
  }
  return result;
}
int clamp_byte(int value) {
  if ((0) < (value)) {
    if ((value) < (255)) {
      return value;
    } else {
      return 255;
    }
  } else {
    return 0;
  }
}
void emit_color(int r, int g, int b, bool bg) {
  r = clamp_byte(r);
  g = clamp_byte(g);
  b = clamp_byte(b);
  auto ri = best_index(r, COLOR_STEPS, COLOR_STEP_COUNT);
  auto gi = best_index(g, COLOR_STEPS, COLOR_STEP_COUNT);
  auto bi = best_index(b, COLOR_STEPS, COLOR_STEP_COUNT);
  auto rq = COLOR_STEPS[ri];
  auto gq = COLOR_STEPS[gi];
  auto bq = COLOR_STEPS[bi];
  auto gray = static_cast<int>(std::round((
      ((((0.29890f)) * (r))) + ((((0.5870f)) * (g))) + ((((0.1140f)) * (b))))));
  auto gri = best_index(gray, GRAYSCALE_STEPS, GRAYSCALE_STEP_COUNT);
  auto grq = GRAYSCALE_STEPS[gri];
  auto color_index = 0;
  if (((((((0.29890f)) * (sqr(((rq) - (r)))))) +
        ((((0.5870f)) * (sqr(((gq) - (g)))))) +
        ((((0.1140f)) * (sqr(((bq) - (b)))))))) <
      ((((((0.29890f)) * (sqr(((grq) - (r)))))) +
        ((((0.5870f)) * (sqr(((grq) - (g)))))) +
        ((((0.1140f)) * (sqr(((grq) - (b))))))))) {
    color_index = ((16) + (((36) * (ri))) + (((6) * (gi))) + (bi));
  } else {
    color_index = ((232) + (gri));
  }
  if (bg) {
    (std::cout) << ("\x1B[48;5;") << (color_index) << ("m");
  } else {
    (std::cout) << ("\x001B[38;5;") << (color_index) << ("m");
  }
}
void emitCodepoint(int codepoint) {
  if ((codepoint) < (128)) {
    (std::cout) << (static_cast<char>(codepoint));
    return;
  }
  if ((codepoint) < (0x7ff)) {
    (std::cout) << (static_cast<char>(((192) | ((codepoint) >> (6)))));
    (std::cout) << (static_cast<char>(((128) | (((codepoint) & (63))))));
    return;
  }
  if ((codepoint) < (0xffff)) {
    (std::cout) << (static_cast<char>(((224) | ((codepoint) >> (12)))));
    (std::cout) << (static_cast<char>(
        ((128) | ((((codepoint) >> (6)) & (63))))));
    (std::cout) << (static_cast<char>(((128) | (((codepoint) & (63))))));
    return;
  }
  if ((codepoint) < (0x10ffff)) {
    (std::cout) << (static_cast<char>(((240) | ((codepoint) >> (18)))));
    (std::cout) << (static_cast<char>(
        ((128) | ((((codepoint) >> (12)) & (63))))));
    (std::cout) << (static_cast<char>(
        ((128) | ((((codepoint) >> (6)) & (63))))));
    (std::cout) << (static_cast<char>(((128) | (((codepoint) & (63))))));
    return;
  }
  (std::cerr) << ("error");
}
void emit_image(uint8_t *img, int w, int h) {
  auto lastCharData = CharData(0);
  for (int y = 0; (y) <= (((h) - (8))); (y) += (8)) {
    for (int x = 0; (x) <= (((h) - (4))); (y) += (4)) {
      auto charData = createCharData(img, w, h, x, y, 9604, 65535);
      if ((((0) == (x)) || ((charData.bgColor) != (lastCharData.bgColor)))) {
        emit_color(charData.bgColor[0], charData.bgColor[1],
                   charData.bgColor[2], true);
      }
      if ((((0) == (x)) || ((charData.fgColor) != (lastCharData.fgColor)))) {
        emit_color(charData.bgColor[0], charData.bgColor[1],
                   charData.bgColor[2], false);
      }
      emitCodepoint(charData.codePoint);
      lastCharData = charData;
    }
    (std::cout) << ("\x1b[0m") << (std::endl);
  }
}
int main(int argc, char **argv) {

  (std::cout)
      << (std::setw(10))
      << (std::chrono::high_resolution_clock::now().time_since_epoch().count())
      << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
      << (__LINE__) << (" ") << (__func__) << (" ") << ("start") << (" ")
      << (std::setw(8)) << (" argc='") << (argc) << ("'") << (std::setw(8))
      << (" argv[0]='") << (argv[0]) << ("'") << (std::endl) << (std::flush);
  auto fd = ::open("img.raw", O_RDONLY);
  auto const w = 170;
  auto const h = 240;
  auto img =
      reinterpret_cast<uint8_t *>(mmap(nullptr, ((w) * (h) * (3)), PROT_READ,
                                       ((MAP_FILE) | (MAP_SHARED)), fd, 0));
  emit_image(img, w, h);
  munmap(img, ((w) * (h) * (3)));
  ::close(fd);
  return 0;
}