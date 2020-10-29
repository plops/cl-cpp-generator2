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
inline int clamp_byte(int value);
void emit_color(int r, int g, int b, bool bg);
int best_index(int value, array(const int) data[], int count);
void emit_image(uint8_t *img, int w, int h);
int main(int argc, char **argv);
#endif