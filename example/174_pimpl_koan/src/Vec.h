//
// Created by martin on 4/5/25.
//

#ifndef VEC_H
#define VEC_H

#include "IContainer.h"
#include <vector>

class Vec : public IContainer
{
public:
    explicit Vec(int n);
    void insert(float) override;
    void resize(int) override;
    float operator[](int idx) override;

private:
    std::vector<float> v;
};



#endif //VEC_H
