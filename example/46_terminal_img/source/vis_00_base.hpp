#ifndef VIS_00_BASE_H
#define VIS_00_BASE_H
#include "utils.h"
;
#include "globals.h"
;
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
;
// header
;
class CharData {
public:
  CharData(int codepoint);
  std::array<int, 3> fgColor = std::array<int, 3>{0, 0, 0};
  std::array<int, 3> bgColor = std::array<int, 3>{0, 0, 0};
  int codePoint;
};
CharData createCharData(uint8_t *img, int w, int h, int x0, int y0,
                        int codepoint, int pattern);
float sqr(float x);
int best_index(int value, const int data[], int count);
inline int clamp_byte(int value);
void emit_color(int r, int g, int b, bool bg);
void emitCodepoint(int codepoint);
void emit_image(uint8_t *img, int w, int h);
int main(int argc, char **argv);
#endif