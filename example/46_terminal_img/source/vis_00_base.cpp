
#include "utils.h"

#include "globals.h"

extern State state;
#include <chrono>
#include <fcntl.h>
#include <iostream>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>
using namespace std::chrono_literals;

// implementation
auto l = quasiquote((0(160))(
    15(9601), 255(9602), 4095(9603), 65535(9604), 1048575(9605), 16777215(9606),
    268435455(9607), 4008636142(9610), 3435973836(9612), 2290649224(9614),
    52428(9622), 13107(9623), 3435921408(9624), 3435934515(9626),
    858980352(9629), 859032780(9630), 859045887(9631), 1044480(9473),
    1717986918(9475), 489062(9487), 976486(9491), 1718054912(9495),
    1718542336(9499), 1718056550(9507), 1718543974(9515), 1046118(9523),
    1718611968(9531), 1718613606(9547), 835584(9592), 417792(9593),
    208896(9594), 417792(9595), 106956384(9551), 983040(9472), 61440(9472),
    1145324612(9474), 572662306(9474), 917504(9588), 57344(9588),
    1145307136(9589), 572653568(9589), 196608(9590), 12288(9590), 17476(9591),
    8738(9591), 1145324612(9122), 572662306(9125), 251658240(9146),
    15728640(9147), 3840(9148), 240(9149), 417792(9642)));
uint8_t *img;
int main(int argc, char **argv) {

  (std::cout)
      << (std::setw(10))
      << (std::chrono::high_resolution_clock::now().time_since_epoch().count())
      << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
      << (__LINE__) << (" ") << (__func__) << (" ") << ("start") << (" ")
      << (std::setw(8)) << (" argc='") << (argc) << ("'") << (std::setw(8))
      << (" argv[0]='") << (argv[0]) << ("'") << (std::endl) << (std::flush);
  auto fd = ::open("img.raw", O_RDONLY);
  auto img = reinterpret_cast<uint8_t *>(
      mmap(nullptr, ((170) * (240) * (3)), PROT_READ,
           ((MAP_FILE) | (MAP_SHARED)), fd, 0));
  munmap(img, ((170) * (240) * (3)));
  ::close(fd);
  return 0;
}