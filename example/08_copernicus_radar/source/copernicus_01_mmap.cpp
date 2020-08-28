
#include "utils.h"

#include "globals.h"

#include "proto2.h"

extern State state;
#include <cassert>
#include <cstdio>
#include <fcntl.h>
#include <iostream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
size_t get_filesize(const char *filename) {
  struct stat st;
  stat(filename, &st);
  return st.st_size;
}
void destroy_mmap() {
  auto rc = munmap(state._mmap_data, state._mmap_filesize);
  if (!((0) == (rc))) {
    std::setprecision(3);
    (std::cout) << (std::setw(10))
                << (((std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count()) -
                     (state._start_time)))
                << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                << (__func__) << (" ") << ("fail munmap") << (" ")
                << (std::setw(8)) << (" rc=") << (rc) << (std::endl);
  }
  assert((0) == (rc));
}
void init_mmap(const char *filename) {
  auto filesize = get_filesize(filename);
  auto fd = open(filename, O_RDONLY, 0);
  std::setprecision(3);
  (std::cout) << (std::setw(10))
              << (((std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count()) -
                   (state._start_time)))
              << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
              << (__func__) << (" ") << ("size") << (" ") << (std::setw(8))
              << (" filesize=") << (filesize) << (std::setw(8))
              << (" filename=") << (filename) << (std::endl);
  if ((-1) == (fd)) {
    std::setprecision(3);
    (std::cout) << (std::setw(10))
                << (((std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count()) -
                     (state._start_time)))
                << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                << (__func__) << (" ") << ("fail open") << (" ")
                << (std::setw(8)) << (" fd=") << (fd) << (std::setw(8))
                << (" filename=") << (filename) << (std::endl);
  }
  assert((-1) != (fd));
  auto data = mmap(NULL, filesize, PROT_READ, MAP_PRIVATE, fd, 0);
  if ((MAP_FAILED) == (data)) {
    std::setprecision(3);
    (std::cout) << (std::setw(10))
                << (((std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count()) -
                     (state._start_time)))
                << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                << (__func__) << (" ") << ("fail mmap") << (" ")
                << (std::setw(8)) << (" data=") << (data) << (std::endl);
  }
  assert((MAP_FAILED) != (data));
  state._mmap_filesize = filesize;
  state._mmap_data = data;
}