
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
uint8_t *img;
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
      auto charData = createCharData(img, w, h, x, y, 0x2584, 0xFFFF);
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
  auto const w = 512;
  auto const h = 285;
  auto img =
      reinterpret_cast<uint8_t *>(mmap(nullptr, ((w) * (h) * (3)), PROT_READ,
                                       ((MAP_FILE) | (MAP_SHARED)), fd, 0));
  emit_image(img, w, h);
  munmap(img, ((w) * (h) * (3)));
  ::close(fd);
  return 0;
}