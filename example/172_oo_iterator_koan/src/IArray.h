//
// Created by martin on 3/26/25.
//

#ifndef IARRAY_H
#define IARRAY_H

template <typename T>
class IArray {
  public:
  virtual ~IArray() noexcept(false) = default;
  virtual T aref(size_t index) = 0;
  virtual T* data() = 0;
  virtual size_t size() = 0;
};

#endif //IARRAY_H
