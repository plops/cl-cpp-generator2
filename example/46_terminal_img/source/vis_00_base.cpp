
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
#include "vis_01_complex.hpp"
uint8_t *img;
CharData::CharData(int codepoint) : codePoint(codepoint) {}
CharData createCharData_simple(uint8_t *img, int w, int h, int x0, int y0,
                               int codepoint, int pattern) {
  auto result = CharData(codepoint);
  const int fg_count = ((4) * (4));
  const int bg_count = ((4) * (4));
  auto mask = 0x80000000;
  auto *avg = result.bgColor.data();
  for (auto y = 0; (y) < (4); (y) += (1)) {
    for (auto x = 0; (x) < (4); (x) += (1)) {
      for (auto i = 0; (i) < (3); (i) += (1)) {
        (avg[i]) +=
            (img[((i) + (((3) * (((x0) + (x) + (((w) * (((y0) + (y))))))))))]);
      }
    }
  }
  auto *avg1 = result.fgColor.data();
  for (auto y = 0; (y) < (4); (y) += (1)) {
    for (auto x = 0; (x) < (4); (x) += (1)) {
      for (auto i = 0; (i) < (3); (i) += (1)) {
        (avg1[i]) += (img[(
            (i) + (((3) * (((x0) + (x) + (((w) * (((y0) + (y) + (4))))))))))]);
      }
    }
  }
  // average color for each bucket
  ;
  for (auto i = 0; (i) < (3); (i) += (1)) {
    result.bgColor[i] = ((result.bgColor[i]) / (bg_count));
  }
  for (auto i = 0; (i) < (3); (i) += (1)) {
    result.fgColor[i] = ((result.fgColor[i]) / (fg_count));
  }
  return result;
}
float sqr(float x) { return ((x) * (x)); }
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
  if (bg) {
    (std::cout) << ("\x1b[48;2;") << (r) << (";") << (g) << (";") << (b)
                << ("m");
  } else {
    (std::cout) << ("\x1b[38;2;") << (r) << (";") << (g) << (";") << (b)
                << ("m");
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
    for (int x = 0; (x) <= (((w) - (4))); (x) += (4)) {
      auto charData = findCharData(img, w, h, x, y);
      if ((((0) == (x)) || ((charData.bgColor) != (lastCharData.bgColor)))) {
        emit_color(charData.bgColor[0], charData.bgColor[1],
                   charData.bgColor[2], true);
      }
      if ((((0) == (x)) || ((charData.fgColor) != (lastCharData.fgColor)))) {
        emit_color(charData.fgColor[0], charData.fgColor[1],
                   charData.fgColor[2], false);
      }
      emitCodepoint(charData.codePoint);
      lastCharData = charData;
    }
    (std::cout) << ("\x1b[0m") << (std::endl);
  }
}
int main(int argc, char **argv) {
  auto fd = ::open("img.raw", O_RDONLY);
  auto const w = 300;
  auto const h = 200;
  auto img =
      reinterpret_cast<uint8_t *>(mmap(nullptr, ((w) * (h) * (3)), PROT_READ,
                                       ((MAP_FILE) | (MAP_SHARED)), fd, 0));
  for (auto i = 0; (i) < (10); (i) += (1)) {
    emit_image(img, w, h);
  }
  munmap(img, ((w) * (h) * (3)));
  ::close(fd);
  return 0;
}