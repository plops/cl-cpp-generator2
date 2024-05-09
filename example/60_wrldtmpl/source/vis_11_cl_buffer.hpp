#ifndef VIS_11_CL_BUFFER_H
#define VIS_11_CL_BUFFER_H
#include "utils.h"
;
#include "globals.h"
;
// header
;
class Buffer {
public:
  enum {
    (DEFAULT) = (0),
    (TEXTURE) = (8),
    (TARGET) = (16),
    (READONLY) = (1),
    (WRITEONLY) = (2)
  };
  Buffer();
  Buffer(unsigned int N, unsigned int tt = DEFAULT, void *ptr = 0);
  ~Buffer();
  cl_mem *GetDevicePtr();
  unsigned int *GetHostPtr();
  void CopyToDevice(bool blocking = true);
  void CopyToDevice2(bool blocking, cl_event *e = 0, const size_t s = 0);
  void CopyFromDevice(bool blocking = true);
  void CopyTo(Buffer *buffer);
  void Clear();
  unsigned int *hostBuffer;
  cl_mem deviceBuffer, pinnedBuffer;
  unsigned int type, size, textureID;
  bool ownData;
};
#endif