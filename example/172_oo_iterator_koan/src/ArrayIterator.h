//
// Created by martin on 3/26/25.
//

#ifndef ARRAYITERATOR_H
#define ARRAYITERATOR_H

template <typename T>
class ArrayIterator {
public:
    using ValueType     = T;
    using PointerType   = T*;
    using ReferenceType = T&;
    ArrayIterator(PointerType ptr);
    ReferenceType     operator*() const;
    ArrayIterator<T>& operator++();
    bool              operator!=(const ArrayIterator<T>& other) const;

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
ArrayIterator<T>& ArrayIterator<T>::operator++() {
    ++current;
    return *this;
}
template <typename T>
bool ArrayIterator<T>::operator!=(const ArrayIterator<T>& other) const {
    return current != other.current;
}
#endif // ARRAYITERATOR_H
