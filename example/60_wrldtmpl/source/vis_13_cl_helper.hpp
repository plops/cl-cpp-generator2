#ifndef VIS_13_CL_HELPER_H
#define VIS_13_CL_HELPER_H
#include "utils.h"
;
#include "globals.h"
;
// header

#define CHECKCL(r) CheckCL(r, __FILE__, __LINE__);

bool CheckCL(cl_int result, const char *file, int line);

cl_device_id getFirstDevice(cl_context context);

cl_int getPlatformID(cl_platform_id *platform);
#endif