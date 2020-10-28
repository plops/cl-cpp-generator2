
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