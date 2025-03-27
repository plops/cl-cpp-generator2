//
// Created by martin on 3/27/25.
//

#ifndef IARRAYITERATOR_H
#define IARRAYITERATOR_H

template <typename T>
class IArrayIterator {
public:
    using ValueType                                                             = T;
    using PointerType                                                           = T*;
    using ReferenceType                                                         = T&;
    virtual ~IArrayIterator() noexcept(false)                                   = default;
    virtual ReferenceType      operator*() const                                = 0;
    virtual IArrayIterator<T>& operator++()                                     = 0;
    virtual PointerType        getCurrent() const                               = 0;
    virtual void               setCurrent(PointerType)                          = 0;
    virtual bool               operator!=(const IArrayIterator<T>& other) const = 0;
};


#endif // IARRAYITERATOR_H
