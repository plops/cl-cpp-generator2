#ifndef VIS_09_FILE_HELPER_H
#define VIS_09_FILE_HELPER_H
#include "utils.h"
;
#include "globals.h"
;
// header
;

bool FileExists(const char *f);

bool RemoveFile(const char *f);

uint FileSize(string f);

string TextFileRead(const char *f);
#endif