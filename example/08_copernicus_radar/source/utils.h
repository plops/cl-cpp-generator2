#ifndef UTILS_H

#define UTILS_H

#include <array>
#include <iomanip>
#include <iostream>
#include <vector>

struct sequential_bit_t {
  size_t current_bit_count;
  uint8_t *data;
};
typedef struct sequential_bit_t sequential_bit_t;
;

#endif
