//
// Created by martin on 4/5/25.
//

#ifndef ICONTAINER_H
#define ICONTAINER_H

class IContainer
{
public:
    virtual ~IContainer() = default;
    virtual void insert(float) =0;
    virtual void resize(int) =0;
    virtual int size() =0;
    virtual float& operator[](int idx) =0;
};

#endif //ICONTAINER_H
