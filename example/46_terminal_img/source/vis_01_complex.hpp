#ifndef VIS_01_COMPLEX_H
#define VIS_01_COMPLEX_H
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
CharData createCharData(uint8_t *img, int w, int h, int x0, int y0,
                        int codepoint, int pattern);
CharData findCharData(uint8_t *img, int w, int h, int x0, int y0);
#endif