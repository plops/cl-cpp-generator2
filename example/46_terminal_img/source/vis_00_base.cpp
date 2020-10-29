
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

// bla
uint8_t *img;
const int COLOR_STEPS[6] = {0x0, 0x5F, 0x87, 0xAF, 0xD7, 0xFF};
const int COLOR_STEP_COUNT = 6;
const int GRAYSCALE_STEP_COUNT = 24;
const int GRAYSCALE_STEPS[24] = {
    0x8,  0x12, 0x1C, 0x26, 0x30, 0x3A, 0x44, 0x4E, 0x58, 0x62, 0x6C, 0x76,
    0x80, 0x8A, 0x94, 0x9E, 0xA8, 0xB2, 0xBC, 0xC6, 0xD0, 0xDA, 0xE4, 0xEE};
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
  auto gri = best_index(grey, GRAYSCALE_STEPS, GRAYSCALE_STEP_COUNT);
  auto grq = GRAYSCALE_STEPS[gri];
}
int best_index(int value, array(const int) data[], int count) {
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
void emit_image(uint8_t *img, int w, int h) {
  auto lastCharData = CharData();
  for (int y = 0; (y) <= (((h) - (8))); (y) += (8)) {
    for (int x = 0; (x) <= (((h) - (4))); (y) += (4)) {
      auto charData = createCharData(img, w, h, x, y, 9604, 65535);
      ;
    }
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
  munmap(img, ((w) * (h) * (3)));
  ::close(fd);
  return 0;
}