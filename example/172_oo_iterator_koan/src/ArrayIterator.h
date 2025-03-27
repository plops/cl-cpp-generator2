//
// Created by martin on 3/26/25.
//

#ifndef ARRAYITERATOR_H
#define ARRAYITERATOR_H
#include "IArrayIterator.h"

template <typename T>
class ArrayIterator : public IArrayIterator<T> {
public:
    using ValueType     = T;
    using PointerType   = T*;
    using ReferenceType = T&;
    ArrayIterator(PointerType ptr);
    ReferenceType     operator*() const override;
    IArrayIterator<T>& operator++() override;
    PointerType       getCurrent() const override;
    void               setCurrent(PointerType) override;
    bool              operator!=(const IArrayIterator<T>& other) const override;

private:
    PointerType current;
};

template <typename T>
ArrayIterator<T>::ArrayIterator(PointerType ptr) : current{ptr} {}
template <typename T>
typename ArrayIterator<T>::ReferenceType ArrayIterator<T>::operator*() const {
    return *current;
}
template <typename T>
IArrayIterator<T>& ArrayIterator<T>::operator++() {
    ++current;
    return *this;
}
template <typename T>
typename ArrayIterator<T>::PointerType ArrayIterator<T>::getCurrent() const {
    return current;
}
template <typename T>
void ArrayIterator<T>::setCurrent(PointerType ptr) {
    current = ptr;
}

template <typename T>
bool ArrayIterator<T>::operator!=(const IArrayIterator<T>& other) const {
    return current != other.getCurrent();
}
#endif // ARRAYITERATOR_H
