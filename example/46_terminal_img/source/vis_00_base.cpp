
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
  auto r = clamp_byte(r);
  auto g = clamp_byte(g);
  auto b = clamp_byte(b);
  ((1) + (2));
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