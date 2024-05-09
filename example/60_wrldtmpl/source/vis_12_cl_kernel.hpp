#ifndef VIS_12_CL_KERNEL_H
#define VIS_12_CL_KERNEL_H
#include "utils.h"
;
#include "globals.h"
;
// header
;
class Kernel {
  friend class Buffer;

public:
  Kernel(char *file, char *entryPoint);
  Kernel(cl_program &existingProgram, char *entryPoint);
  ~Kernel();
  cl_kernel &GetKernel();
  cl_program &GetProgram();
  static cl_command_queue &GetQueue();
  static cl_command_queue &GetQueue2();
  static cl_context &GetContext();
  static cl_device_id &GetDevice();
  void Run(cl_event *eventToWaitFor = 0, cl_event *eventToSet = 0);
  void Run(cl_mem *buffers, int count = 1, cl_event *eventToWaitFor = 0,
           cl_event *eventToSet = 0, cl_event *acq = 0, cl_event *rel = 0);
  void Run(Buffer *buffer, const int2 tileSize, cl_event *eventToWaitFor = 0,
           cl_event *eventToSet = 0, cl_event *acq = 0, cl_event *rel = 0);
  void Run(const size_t count, const int2 localSize = make_int2(32, 2),
           cl_event *eventToWaitFor = 0, cl_event *eventToSet = 0);
  void Run2D(const int2 count, const int2 lsize, cl_event *eventToWaitFor = 0,
             cl_event *eventToSet = 0);
  void SetArgument(int idx, cl_mem *buffer);
  void SetArgument(int idx, Buffer *buffer);
  void SetArgument(int idx, float value);
  void SetArgument(int idx, int value);
  void SetArgument(int idx, float2 value);
  void SetArgument(int idx, float3 value);
  void SetArgument(int idx, float4 value);
  bool InitCL();

private:
  cl_kernel kernel;
  cl_mem vbo_cl;
  cl_program program;
  bool arg0set = false;
  inline static cl_device_id device;
  inline static cl_context context;
  inline static cl_command_queue queue, queue2;
  inline static char *log = 0;

public:
  inline static bool candoInterop = false;
};
#endif