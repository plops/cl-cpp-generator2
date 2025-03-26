//
// Created by martin on 3/26/25.
//

#ifndef GENERALARRAY_H
#define GENERALARRAY_H
#include "IArray.h"
#include <memory>
template <typename T>
class GeneralArray : public IArray<T> {
public:
T         aref(size_t index) override;
    T*        data() override;
    size_t    size() override;
    decltype(pimpl_->begin()) begin() override;
    // TIterator end() override;

private:
    std::unique_ptr<IArray<T>> pimpl_;
};



#endif //GENERALARRAY_H
