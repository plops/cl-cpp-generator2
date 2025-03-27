//
// Created by martin on 3/26/25.
//

#ifndef GENERALARRAY_H
#define GENERALARRAY_H
#include <memory>
#include "IArray.h"
template <typename T, class TArray, class TIterator>
class GeneralArray : public IArray<T, TIterator> {
public:
    GeneralArray();
    T         aref(size_t index) override;
    T*        data() override;
    size_t    size() override;
    TIterator begin() override;
    TIterator end() override;

private:
    std::unique_ptr<IArray<T, TIterator>> pimpl_;
};


template <typename T, class TArray, class TIterator>
GeneralArray<T, TArray, TIterator>::GeneralArray() : pimpl_{std::make_unique<TArray>()} {}
template <typename T, class TArray,  class TIterator>
T GeneralArray<T, TArray, TIterator>::aref(size_t index) { return pimpl_->aref(index); }
template <typename T, class TArray,  class TIterator>
T* GeneralArray<T, TArray, TIterator>::data() { return pimpl_->data(); }
template <typename T, class TArray,  class TIterator>
size_t GeneralArray<T, TArray, TIterator>::size() { return pimpl_->size(); }
template <typename T, class TArray,  class TIterator>
TIterator GeneralArray<T, TArray, TIterator>::begin() { return pimpl_->begin(); }
template <typename T, class TArray,  class TIterator>
TIterator GeneralArray<T, TArray, TIterator>::end() { return pimpl_->end(); }
#endif // GENERALARRAY_H
