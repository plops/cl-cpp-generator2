#ifndef VIS_01_BASE_H
#define VIS_01_BASE_H
#include "utils.h"
;
#include "globals.h"
;
#include <iostream>
#include <chrono>
#include <thread>
;
#include "mkl.h"
;
void init_matrix (double* a, int m, int n)  ;  
std::chrono::high_resolution_clock::time_point get_time ()  ;  
int main (int argc, char** argv)  ;  
#endif