//
// Created by martin on 3/26/25.
//

#ifndef ARRAYITERATOR_H
#define ARRAYITERATOR_H

#include "IArrayIterator.h"

template <typename T>
class ArrayIterator : public IArrayIterator<T> {
public:
    ArrayIterator(IArrayIterator<T>::PointerType ptr);
    IArrayIterator<T>::ReferenceType operator*() const override;
    IArrayIterator<T>&               operator++() override;
    bool                             operator!=(const IArrayIterator<T>& other) const override;

private:
    IArrayIterator<T>::PointerType current;
};

template <typename T>
ArrayIterator<T>::ArrayIterator(IArrayIterator<T>::PointerType ptr) : current{ptr} {}
template <typename T>
typename IArrayIterator<T>::ReferenceType ArrayIterator<T>::operator*() const { return *current; }
template <typename T>
IArrayIterator<T>& ArrayIterator<T>::operator++() {
    ++current;
    return *this;
}
template <typename T>
bool ArrayIterator<T>::operator!=(const IArrayIterator<T>& other) const {
    return current != other.current;
}
#endif // ARRAYITERATOR_H
