//
// Created by martin on 4/5/25.
//

#ifndef WIDGET_H
#define WIDGET_H

#include <memory>
#include "IWidget.h"

class SharedWidget : public IWidget
{
public:
    explicit SharedWidget(int start);
    int add(int a, int b) override;

private:
    struct Impl;
    std::shared_ptr<Impl> pImpl;
};


#endif //WIDGET_H
