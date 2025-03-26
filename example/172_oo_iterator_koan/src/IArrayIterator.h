//
// Created by martin on 3/26/25.
//

#ifndef IARRAYITERATOR_H
#define IARRAYITERATOR_H
#include <cstddef>

template <typename T>
class IArrayIterator {
public:
    using ValueType      = T;
    // using DifferenceType = std::ptrdiff_t;
    using PointerType    = T*;
    using ReferenceType  = T&;

    virtual ReferenceType operator*() const = 0; // Dereference operator to access the element
    virtual IArrayIterator& operator++() = 0; // Pre-increment
    virtual bool operator!=(const IArrayIterator& other) const = 0; // Inequality comparison
};

#endif // IARRAYITERATOR_H
