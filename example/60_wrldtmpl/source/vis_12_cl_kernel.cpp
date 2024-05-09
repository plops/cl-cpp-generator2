
#include "utils.h"

#include "globals.h"

extern State state;
// implementation

Kernel::Kernel(char *file, char *entryPoint) {
  nil;
  auto size{static_cast<size_t>(0)};
  auto pos{size};
  auto err{static_cast<cl_int>(0)};
  auto csText{TextFileRead(file)};
  auto incLines{0};
  if (!(csText.size())) {
    {

      auto lock{std::unique_lock<std::mutex>(state._stdout_mutex)};
      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ") << ("file not found") << (" ") << (std::setw(8))
                  << (" file='") << (file) << ("'") << (std::endl)
                  << (std::flush);
    }
  }
  while (true) {
    (pos) = (csText.find("#include"));
    if ((pos) == (string::npos)) {
      break;
    }
    string tmp;
    if ((0) < (pos)) {
      (tmp) = (csText.substr(0, (pos) - (1)));
    }
    (pos) = (csText.find("\""));
    if ((pos) == (string::npos)) {
      {

        auto lock{std::unique_lock<std::mutex>(state._stdout_mutex)};
        (std::cout) << (std::setw(10))
                    << (std::chrono::high_resolution_clock::now()
                            .time_since_epoch()
                            .count())
                    << (" ") << (std::this_thread::get_id()) << (" ")
                    << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                    << (" ")
                    << ("expected double quote after #include in shader")
                    << (" ") << (std::endl) << (std::flush);
      }
    }
    auto end{csText.find("\"", (pos) + (1))};
    if ((end) == (string::npos)) {
      {

        auto lock{std::unique_lock<std::mutex>(state._stdout_mutex)};
        (std::cout) << (std::setw(10))
                    << (std::chrono::high_resolution_clock::now()
                            .time_since_epoch()
                            .count())
                    << (" ") << (std::this_thread::get_id()) << (" ")
                    << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                    << (" ")
                    << ("expected second double quote after #include in shader")
                    << (" ") << (std::endl) << (std::flush);
      }
    }
    auto file{csText.substr((pos) + (1), (end) - (pos) - (1))};
    auto incText{TextFileRead(file.c_str())};
    auto p{incText.c_str()};
    while (p) {
      (incLines)++;
      (p) = (strstr((p) + (1), "\n"));
    }
    (incLines) -= (2);
    (tmp) += (incText);
    (tmp) += (csText.substr((end) + (1), string::npos));
    (csText) = (tmp);
  }
  auto source{csText.c_str()};
  auto size{strlen(source)};
  auto program{clCreateProgramWithSource(
      context, 1, static_cast<const char **>(&source), &size, &err)};
  CHECKCL(err);
  (err) = (clBuildProgram(
      program, 0, nullptr,
      "-cl-fast-relaxed-math -cl-mad-enable  -cl-denorms-are-zero "
      "-cl-no-signed-zeros -cl-unsafe-math-optimizations",
      nullptr, nullptr));
  if (!((CL_SUCCESS) == (err))) {
    if (!(log)) {
      (log) = (new (char)[((256) * (1024))]);
    }
    ((log)[(0)]) = (0);
    clGetProgramBuildInfo(program, getFirstDevice(context),
                          CL_PROGRAM_BUILD_LOG, (100) * (1024), log, nullptr);
    ((log)[(2048)]) = (0);
    {

      auto lock{std::unique_lock<std::mutex>(state._stdout_mutex)};
      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ") << ("build error") << (" ") << (std::setw(8))
                  << (" log='") << (log) << ("'") << (std::endl)
                  << (std::flush);
    }
  }
  (kernel) = (clCreateKernel(program, entryPoint, &err));
  nil;
}
Kernel::Kernel(cl_program &existingProgram, char *entryPoint) {
  nil;
  auto err{static_cast<cl_int>()};
  (program) = (existingProgram);
  (kernel) = (clCreateKernel(program, entryPoint, &err));
  if (!(kernel)) {
    {

      auto lock{std::unique_lock<std::mutex>(state._stdout_mutex)};
      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ") << ("create kernel failed: entry point not found.")
                  << (" ") << (std::endl) << (std::flush);
    }
  }
  CHECKCL(err);
  nil;
}
Kernel::~Kernel() {
  nil;
  if (kernel) {
    clReleaseKernel(kernel);
  }
  if (program) {
    clReleaseProgram(program);
  }
  nil;
}
cl_kernel &Kernel::GetKernel() {
  nil;
  return kernel;
  nil;
}
cl_program &Kernel::GetProgram() {
  nil;
  return program;
  nil;
}
static cl_command_queue &Kernel::GetQueue() {
  nil;
  return queue;
  nil;
}
static cl_command_queue &Kernel::GetQueue2() {
  nil;
  return queue2;
  nil;
}
static cl_context &Kernel::GetContext() {
  nil;
  return context;
  nil;
}
static cl_device_id &Kernel::GetDevice() {
  nil;
  return device;
  nil;
}
void Kernel::Run(cl_event *eventToWaitFor, cl_event *eventToSet) {
  nil;
  glFinish();
  auto err{clEnqueueNDRangeKernel(queue, kernel, 2, 0, workSize, localSize,
                                  (eventToWaitFor) ? (1) : (0), eventToWaitFor,
                                  eventToSet)};
  CHECKCL(err);
  clFinish(queue);
  nil;
}
void Kernel::Run(cl_mem *buffers, int count, cl_event *eventToWaitFor,
                 cl_event *eventToSet, cl_event *acq, cl_event *rel) {
  nil;
  auto err{static_cast<cl_int>(0)};
  if (Kernel::candoInterop) {
    CHECKCL((err) =
                (clEnqueueAcquireGLObjects(queue, count, buffers, 0, 0, acq)));
    CHECKCL((err) = (clEnqueueNDRangeKernel(
                queue, kernel, 2, 0, workSize, localSize,
                (eventToWaitFor) ? (1) : (0), eventToWaitFor, eventToSet)));
    CHECKCL((err) =
                (clEnqueuReleaseGLObjects(queue, count, buffers, 0, 0, rel)));
  } else {
    CHECKCL((err) = (clEnqueueNDRangeKernel(
                queue, kernel, 2, 0, workSize, localSize,
                (eventToWaitFor) ? (1) : (0), eventToWaitFor, eventToSet)));
  }
  nil;
}
void Kernel::Run(Buffer *buffer, const int2 tileSize, cl_event *eventToWaitFor,
                 cl_event *eventToSet, cl_event *acq, cl_event *rel) {
  nil;
  if (!(arg0set)) {
    {

      auto lock{std::unique_lock<std::mutex>(state._stdout_mutex)};
      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ")
                  << ("kernel expects at least one argument, none set.")
                  << (" ") << (std::endl) << (std::flush);
    }
  }
  auto err{static_cast<cl_int>(0)};
  if (Kernel::candoInterop) {
    size_t localSize[2]{
        {static_cast<size_t>(tileSize.x), static_cast<size_t>(tileSize.y)}};
    auto count{1};
    CHECKCL((err) = (clEnqueueAcquireGLObjects(
                queue, count, buffer->GetDevicePtr(), 0, 0, acq)));
    CHECKCL((err) = (clEnqueueNDRangeKernel(
                queue, kernel, 2, 0, workSize, localSize,
                (eventToWaitFor) ? (1) : (0), eventToWaitFor, eventToSet)));
    CHECKCL((err) =
                (clEnqueuReleaseGLObjects(queue, count, buffers, 0, 0, rel)));
  } else {
    CHECKCL((err) = (clEnqueueNDRangeKernel(
                queue, kernel, 2, 0, workSize, localSize,
                (eventToWaitFor) ? (1) : (0), eventToWaitFor, eventToSet)));
  }
  nil;
}
void Kernel::Run(const size_t count, const int2 localSize,
                 cl_event *eventToWaitFor, cl_event *eventToSet) {
  nil;
  auto err{static_cast<cl_int>(0)};
  CHECKCL((err) = (clEnqueueNDRangeKernel(
              queue, kernel, 1, 0, &count,
              ((0) == (localSize)) ? (0) : (&localSize),
              (eventToWaitFor) ? (1) : (0), eventToWaitFor, eventToSet)));
  nil;
}
void Kernel::Run2D(const int2 count, const int2 lsize, cl_event *eventToWaitFor,
                   cl_event *eventToSet) {
  nil;
  auto localSize{{static_cast<size_t>(lsize.x), static_cast<size_t>(lsize.y)}};
  auto workSize{{static_cast<size_t>(count.x), static_cast<size_t>(count.y)}};
  auto err{static_cast<cl_int>(0)};
  CHECKCL((err) = (clEnqueueNDRangeKernel(
              queue, kernel, 2, 0, workSize, localSize,
              (eventToWaitFor) ? (1) : (0), eventToWaitFor, eventToSet)));
  nil;
}
void Kernel::SetArgument(int idx, cl_mem *buffer) {
  nil;
  clSetKernelArg(kernel, idx, sizeof(cl_mem), buffer);
  (arg0set) = ((arg0set) || ((0) == (idx)));
  nil;
}
void Kernel::SetArgument(int idx, Buffer *buffer) {
  nil;
  clSetKernelArg(kernel, idx, sizeof(cl_mem), buffer->GetDevicePtr());
  (arg0set) = ((arg0set) || ((0) == (idx)));
  nil;
}
void Kernel::SetArgument(int idx, float value) {
  nil;
  clSetKernelArg(kernel, idx, sizeof(float), &value);
  (arg0set) = ((arg0set) || ((0) == (idx)));
  nil;
}
void Kernel::SetArgument(int idx, int value) {
  nil;
  clSetKernelArg(kernel, idx, sizeof(int), &value);
  (arg0set) = ((arg0set) || ((0) == (idx)));
  nil;
}
void Kernel::SetArgument(int idx, float2 value) {
  nil;
  clSetKernelArg(kernel, idx, sizeof(float2), &value);
  (arg0set) = ((arg0set) || ((0) == (idx)));
  nil;
}
void Kernel::SetArgument(int idx, float3 value) {
  nil;
  clSetKernelArg(kernel, idx, sizeof(float3), &value);
  (arg0set) = ((arg0set) || ((0) == (idx)));
  nil;
}
void Kernel::SetArgument(int idx, float4 value) {
  nil;
  clSetKernelArg(kernel, idx, sizeof(float4), &value);
  (arg0set) = ((arg0set) || ((0) == (idx)));
  nil;
}
bool Kernel::InitCL() {
  nil;
  cl_platform_id platform;
  auto err{static_cast<cl_int>(0)};
  auto devCount{static_cast<cl_uint>(0)};
  auto devices{static_cast<cl_device_id *>(nullptr)};
  if (!(CHECKCL((err) = (getPlatformID(&platform))))) {
    return false;
  }
  if (!(CHECKCL((err) = (clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0,
                                        nullptr, &devCount))))) {
    return false;
  }
  (devices) = (new (cl_device_id)[(devCount)]);
  if (!(CHECKCL((err) = (clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devCount,
                                        devices, nullptr))))) {
    return false;
  }
  auto deviceUsed{static_cast<uint>(-1)};
  auto endDev{static_cast<uint>((devCount) - (1))};
  for (auto i = 0; (i) < (endDev); (i) += (1)) {
    auto extensionSize{static_cast<size_t>(0)};
    CHECKCL((err) = (clGetDeviceInfo((devices)[(i)], CL_DEVICE_EXTENSIONS, 0,
                                     nullptr, &extensionSize)));
    if ((0) < (extensionSize)) {
      auto extensions{static_cast<char *>(malloc(exensionSize))};
      CHECKCL((err) =
                  (clGetDeviceInfo((devices)[(i)], CL_DEVICE_EXTENSIONS,
                                   extensionSize, extensions, &extensionSize)));
      string devices(extensionsions);
      free(extensinos);
      auto o{static_cast<size_t>(0)};
      auto s{devices.find(' ', o)};
      while ((s) != (devices.npos)) {
        auto subs{devices.substr(o, (s) - (o))};
        if (!(strcmp("cl_khr_gl_sharing", subs.c_str()))) {
          (deviceUsed) = (1);
          break;
        }
        do {
          (o) = ((s) + (1));
          (s) = (devices.find(' ', o));
        } while ((s) == (o));
      }
      if ((-1) < (deviceUsed)) {
        break;
      }
    }
  }
  cl_context_properties props[]{
      {CL_CONTEXT_PLATFORM, static_cast<cl_context_properties>(platform),
       CL_GL_CONTEXT_KHR,
       static_cast<cl_context_properties>(glfwGetWGLContext(window)), 0}};
  (candoInterop) = (true);
  (context) = (clCreateContext(props, 1, &((devices)[(deviceUsed)]), nullptr,
                               nullptr, &err));
  if ((0) != (err)) {
    {

      auto lock{std::unique_lock<std::mutex>(state._stdout_mutex)};
      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ") << ("no capable opencl device found.") << (" ")
                  << (std::endl) << (std::flush);
    }
  }
  auto device{getFirstDevice(context)};
  if (!(CHECKCL(err))) {
    return false;
  }
  char device_string[1024];
  char device_platform[1024];
  clGetDeviceInfo((devices)[(deviceUsed)], CL_DEVICE_NAME, 1024,
                  &(device_string), nullptr);
  clGetDeviceInfo((devices)[(deviceUsed)], CL_DEVICE_VERSION, 1024,
                  &(device_platform), nullptr);
  {

    auto lock{std::unique_lock<std::mutex>(state._stdout_mutex)};
    (std::cout) << (std::setw(10))
                << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
                << (":") << (__LINE__) << (" ") << (__func__) << (" ")
                << ("device") << (" ") << (std::setw(8)) << (" deviceUsed='")
                << (deviceUsed) << ("'") << (std::setw(8))
                << (" device_string='") << (device_string) << ("'")
                << (std::setw(8)) << (" device_platform='") << (device_platform)
                << ("'") << (std::endl) << (std::flush);
  }
  (queue) = (clCreateCommandQueue(context, (devices)[(deviceUsed)], 0, &err));
  if (!(CHECKCL(err))) {
    return false;
  }
  (queue2) = (clCreateCommandQueue(context, (devices)[(deviceUsed)], 0, &err));
  if (!(CHECKCL(err))) {
    return false;
  }
  delete (devices);
  return true;
  nil;
}