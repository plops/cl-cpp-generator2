#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fcntl.h>
#include <fmt/core.h>
#include <immintrin.h>
#include <iostream>
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <unistd.h>
#define ARRAY_SIZE 1000000

void configure_msr() {
  auto fd = open("/dev/cpu/0/msr", O_RDWR);
  auto msr_perf = 0x38F;
  auto enable_all_counters = 0x00000007000000ffull;
  auto val = uint64_t(0);
  pread(fd, &val, sizeof(val), msr_perf);
  fmt::print("  val='{}'\n", val);
  pwrite(fd, &enable_all_counters, sizeof(enable_all_counters), msr_perf);
  pread(fd, &val, sizeof(val), msr_perf);
  fmt::print("  val='{}'\n", val);

  close(fd);
}

int main(int argc, char **argv) {
  fmt::print("15:09:43 of Saturday, 2023-02-25 (GMT+1)\n");
  configure_msr();
  std::srand(std::time(nullptr));
  auto array = std::array<int, ARRAY_SIZE>();
  for (auto &e : array) {
    e = std::rand();
  };

  // if rdpmc crashes, run this: echo 2 | sudo tee /sys/devices/cpu/rdpmc

  auto cycles = __rdpmc((1 << 30 + 0));

  auto count = 0;
  for (auto &e : array) {
    if (0 == e % 2) {
      count++;
    }
  };

  auto new_cycles = __rdpmc((1 << 30 + 0));

  auto cycles_count = ((new_cycles) - (cycles));
  fmt::print("  cycles_count='{}'\n", cycles_count);

  return 0;
}
