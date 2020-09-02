#ifndef VIS_01_MMAP_H
#define VIS_01_MMAP_H
#include "utils.h"
;
#include "globals.h"
;
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <cstdio>
#include <cassert>
#include <iostream>
#include <thread>
;
size_t get_filesize (const char* filename)  ;  
void destroy_mmap ()  ;  
void init_mmap (const char* filename)  ;  
#endif