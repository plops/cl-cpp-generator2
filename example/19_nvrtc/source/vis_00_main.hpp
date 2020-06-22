#ifndef VIS_00_MAIN_H
#define VIS_00_MAIN_H
#include "utils.h"
;
#include "globals.h"
;
#include <iostream>
#include <chrono>
#include <cstdio>
#include <cassert>
#include <string>
#include <fstream>
#include <thread>
#include <vector>
;
#include <nvrtc.h>
#include <cuda.h>
#include <cuda_runtime.h>
;
#include "vis_03_cu_program.hpp"
;
#include "vis_04_cu_module.hpp"
;
#include "vis_02_cu_device.hpp"
;
#include "vis_01_rtc.hpp"
;
int main ();  
#endif