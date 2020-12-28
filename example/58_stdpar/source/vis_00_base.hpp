#ifndef VIS_00_BASE_H
#define VIS_00_BASE_H
#include "utils.h"
;
#include "globals.h"
;
#include <chrono>
#include <cmath>
#include <execution>
#include <iostream>
#include <thread>
;
#include <thrust/iterator/counting_iterator.h>
;
// header
;
#include <chrono>
#include <cmath>
#include <execution>
#include <iostream>
#include <thread>
#include <thrust/iterator/counting_iterator.h>
;
int jacobi_solver(float *data, int M, int N, float max_diff);
int main(int argc, char **argv);
#endif