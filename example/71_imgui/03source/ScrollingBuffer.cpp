// no preamble

// no implementation preamble

#include "ScrollingBuffer.h"
ScrollingBuffer::ScrollingBuffer(int max_size_)
    : max_size{max_size_}, offset{0} {
  data.reserve(max_size);
}
void ScrollingBuffer::AddPoint(float x, float y) {
  auto v{ImVec2(x, y)};
  if ((data.size()) < (max_size)) {
    data.push_back(v);
  } else {
    ((data)[(offset)]) = (v);
    (offset) = (((offset) + (1)) % (max_size));
  }
}
void ScrollingBuffer::Erase() {
  if ((0) < (data.size())) {
    data.shrink(0);
    (offset) = (0);
  }
}