//
// Created by martin on 3/26/25.
//

#ifndef ARRAY_H
#define ARRAY_H

#include <memory>
#include "IArray.h"
template <typename T, class TIterator = ArrayIterator<T>>
class Array : public IArray<T, TIterator> {
public:
    explicit Array(size_t size);
    ~Array() noexcept(false) override;
    T                aref(size_t index) override;
    T*               data() override;
    size_t           size() override;
    TIterator begin() override;
    TIterator end() override;

private:
    size_t               _size;
    std::unique_ptr<T[]> _data;
};

template <typename T,class TIterator>
Array<T,TIterator>::Array(size_t size) : _size{size}, _data{std::make_unique<T[]>(size)} {}
template <typename T,class TIterator>
Array<T,TIterator>::~Array() noexcept(false) {}
template <typename T,class TIterator>
T Array<T,TIterator>::aref(size_t index) {
    return _data[index];
}
template <typename T,class TIterator>
T* Array<T,TIterator>::data() {
    return _data.get();
}
template <typename T,class TIterator>
size_t Array<T,TIterator>::size() {
    return _size;
}
template <typename T,class TIterator>
TIterator Array<T,TIterator>::begin() {
    return ArrayIterator<T>{_data.get()};
}
template <typename T,class TIterator>
TIterator Array<T,TIterator>::end() {
    return ArrayIterator<T>{_data.get() + _size};
}
#endif // ARRAY_H
