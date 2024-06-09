#include <cstdint>
#include <cstring>
#include <iostream>

float to_float(uint32_t n) {
  n += ((1U << 23) - 1);
  if ((n & (1U << 31)) != 0u) {
    n = n ^ (1U << 31);
  } else {
    n = ~n;
  }
  float f;
  memcpy(&f, &n, 4);
  return f;
}

uint32_t float_to_index(float f) {
  uint32_t n;
  memcpy(&n, &f, sizeof(n));
  if ((n & (1U << 31)) != 0u) {
    n = n ^ (1 << 31);
  } else {
    n = ~n;
  }
  return n - ((1U << 23) - 1);
}

int main(int argc, char **argv) {
  std::cout << ""
            << " to_float(12)='" << to_float(12) << "' "
            << " float_to_index(to_float(12))='" << float_to_index(to_float(12))
            << "' " << std::endl;
  return 0;
}
