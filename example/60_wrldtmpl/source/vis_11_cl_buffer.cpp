
#include "utils.h"

#include "globals.h"

extern State state;
// implementation

Buffer::Buffer() {
  nil;
  // // nothing

  nil;
}
Buffer::Buffer(unsigned int N, unsigned int tt, void *ptr) {
  nil;
  (type) = (tt);
  (ownData) = (false);
  auto rwFlags{CL_MEM_READ_WRITE};
  if (((tt) & (READONLY))) {
    (rwFlags) = (CL_MEM_READ_ONLY);
  }
  if (((tt) & (WRITEONLY))) {
    (rwFlags) = (CL_MEM_WRITE_ONLY);
  }
  if ((0) == (((tt) & ((TEXTURE) || (TARGET))))) {
    (size) = (N);
    (textureID) = (0);
    (deviceBuffer) =
        (clCreateBuffer(Kernel::GetContext(), rwFlags, (size) * (4), 0, 0));
    (hostBuffer) = (static_cast<uint *>(ptr));
  } else {
    (textureID) = (N);
    if (!(Kernel::candoInterop)) {
      {

        auto lock{std::unique_lock<std::mutex>(state._stdout_mutex)};
        (std::cout) << (std::setw(10))
                    << (std::chrono::high_resolution_clock::now()
                            .time_since_epoch()
                            .count())
                    << (" ") << (std::this_thread::get_id()) << (" ")
                    << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                    << (" ") << ("didn't expect to get here") << (" ")
                    << (std::endl) << (std::flush);
      }
    }
    auto err{0};
    if ((TARGET) == (tt)) {
      (deviceBuffer) = (clCreateFromGLTexture(
          Kernel::GetContext(), CL_MEM_WRITE_ONLY, GL_TEXTURE_2D, 0, N, &err));
    } else {
      (deviceBuffer) = (clCreateFromGLTexture(
          Kernel::GetContext(), CL_MEM_READ_ONLY, GL_TEXTURE_2D, 0, N, &err));
      CHECKCL(err);
      (hostBuffer) = (0);
    }
  }
  nil;
}
Buffer::~Buffer() {
  nil;
  if (ownData) {
    delete (hostBuffer);
  }
  nil;
}
cl_mem *Buffer::GetDevicePtr() {
  nil;
  return &deviceBuffer;
  nil;
}
unsigned int *Buffer::GetHostPtr() {
  nil;
  return &hostBuffer;
  nil;
}
void Buffer::CopyToDevice(bool blocking) {
  nil;
  auto err{static_cast<cl_int>(0)};
  CHECKCL((err) =
              (clEnqueueWriteBuffer(Kernel::GetQueue(), deviceBuffer, blocking,
                                    0, (size) * (4), hostBuffer, 0, 0, 0)));
  nil;
}
void Buffer::CopyToDevice2(bool blocking, cl_event *e, const size_t s) {
  nil;
  // this uses the second queue

  auto err{static_cast<cl_int>(0)};
  CHECKCL((err) = (clEnqueueWriteBuffer(
              Kernel::GetQueue2(), deviceBuffer, blocking, 0,
              ((0) == (s)) ? ((size) * (4)) : ((s) * (4)), hostBuffer, 0, 0,
              eventToSet)));
  nil;
}
void Buffer::CopyFromDevice(bool blocking) {
  nil;
  auto err{static_cast<cl_int>(0)};
  CHECKCL((err) =
              (clEnqueueReadBuffer(Kernel::GetQueue(), deviceBuffer, blocking,
                                   0, (size) * (4), hostBuffer, 0, 0, 0)));
  nil;
}
void Buffer::CopyTo(Buffer *buffer) {
  nil;
  clEnqueueCopyBuffer(Kernel::GetQueue(), deviceBuffer, buffer->deviceBuffer, 0,
                      0, (size) * (4), 0, 0, 0);
  nil;
}
void Buffer::Clear() {
  nil;
  auto value{static_cast<uint>(0)};
  auto err{static_cast<cl_int>(0)};
  CHECKCL((err) = (clEnqueueFillBuffer(Kernel::GetQueue(), deviceBuffer, &value,
                                       4, 0, (size) * (4), 0, 0, 0)));
  nil;
}