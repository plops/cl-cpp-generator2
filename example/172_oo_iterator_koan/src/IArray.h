//
// Created by martin on 3/26/25.
//

#ifndef IARRAY_H
#define IARRAY_H
#include "IArrayIterator.h"
template <typename T>
class IArray {
public:
    using ValueType                     = T;
    // using Iterator                      = IArrayIterator<T>;
    virtual ~IArray() noexcept(false)   = default;
    virtual T        aref(size_t index) = 0;
    virtual T*       data()             = 0;
    virtual size_t   size()             = 0;
    // virtual Iterator begin()            = 0;
    // virtual Iterator end()              = 0;
};

#endif // IARRAY_H
