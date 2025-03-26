//
// Created by martin on 3/26/25.
//

#ifndef ARRAY_H
#define ARRAY_H

#include <memory>
#include "ArrayIterator.h"
#include "IArray.h"
#include "IArrayIterator.h"
template <typename T>
class Array : public IArray<T> {
public:
    using ValueType = T;
    explicit Array(size_t size);
    ~Array() noexcept(false) override;
    T      aref(size_t index) override;
    T*     data() override;
    size_t size() override;
    ArrayIterator<T> begin();
    ArrayIterator<T> end();

private:
    size_t               _size;
    std::unique_ptr<T[]> _data;
};

template <typename T>
Array<T>::Array(size_t size) : _size{size},_data {std::make_unique<T[]>(size)} {}
template <typename T>
Array<T>::~Array() noexcept(false) {}
template <typename T>
T Array<T>::aref(size_t index) {return _data[index];}
template <typename T>
T* Array<T>::data() { return _data.get(); }
template <typename T>
size_t Array<T>::size() { return _size; }
template <typename T>
ArrayIterator<T> Array<T>::begin() { return ArrayIterator<T>{_data.get()}; }
template <typename T>
ArrayIterator<T> Array<T>::end() { return ArrayIterator<T>{_data.get() + _size}; }
#endif // ARRAY_H
