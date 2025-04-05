//
// Created by martin on 4/5/25.
//

#ifndef SHARED_WIDGET_H
#define SHARED_WIDGET_H

#include <memory>
#include "IWidget.h"
#include "IContainer.h"

class SharedWidget : public IWidget
{
public:
    explicit SharedWidget(int start, std::unique_ptr<IContainer> container);
    int add(int a, int b) override;
    void insert(float) override;

private:
    struct Impl;
    std::shared_ptr<Impl> pImpl;
};


#endif //SHARED_WIDGET_H
