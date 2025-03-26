//
// Created by martin on 3/26/25.
//

#ifndef VARRAY_H
#define VARRAY_H


#include <memory>
#include <vector>
#include "IArray.h"
template <typename T,class TIterator = std::vector<T>::iterator>
class VArray : public IArray<T,TIterator> {
public:
    explicit VArray(size_t size);
    ~VArray() noexcept(false) override;
    T                        aref(size_t index) override;
    T*                       data() override;
    size_t                   size() override;
    TIterator begin() override;
    TIterator end() override;

private:
    std::vector<T> _data;
};


template <typename T,class TIterator>
VArray<T,TIterator>::VArray(size_t size) : _data(size) {}
template <typename T,class TIterator>
VArray<T,TIterator>::~VArray() noexcept(false) {}
template <typename T,class TIterator>
T VArray<T,TIterator>::aref(size_t index) {
    return _data[index];
}
template <typename T,class TIterator>
T* VArray<T,TIterator>::data() {
    return _data.data();
}
template <typename T,class TIterator>
size_t VArray<T,TIterator>::size() {
    return _data.size();
}
template <typename T,class TIterator>
TIterator VArray<T,TIterator>::begin() {
    return _data.begin();
}
template <typename T,class TIterator>
TIterator VArray<T,TIterator>::end() {
    return _data.end();
}
#endif // VARRAY_H
