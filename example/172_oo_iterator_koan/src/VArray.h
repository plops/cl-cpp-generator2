//
// Created by martin on 3/26/25.
//

#ifndef VARRAY_H
#define VARRAY_H


#include <memory>
#include <vector>
#include "IArray.h"
template <typename T>
class VArray : public IArray<T> {
public:
    explicit VArray(size_t size);
    ~VArray() noexcept(false) override;
    T                        aref(size_t index) override;
    T*                       data() override;
    size_t                   size() override;
    std::vector<T>::iterator begin() override;
    std::vector<T>::iterator end() override;

private:
    std::vector<T> _data;
};


template <typename T>
VArray<T>::VArray(size_t size) : _data(size) {}
template <typename T>
VArray<T>::~VArray() noexcept(false) {}
template <typename T>
T VArray<T>::aref(size_t index) {
    return _data[index];
}
template <typename T>
T* VArray<T>::data() {
    return _data.data();
}
template <typename T>
size_t VArray<T>::size() {
    return _data.size();
}
template <typename T>
std::vector<T>::iterator VArray<T>::begin() {
    return _data.begin();
}
template <typename T>
std::vector<T>::iterator VArray<T>::end() {
    return _data.end();
}
#endif // VARRAY_H
