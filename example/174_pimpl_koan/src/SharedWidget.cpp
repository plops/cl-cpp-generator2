//
// Created by martin on 4/5/25.
//

#include "SharedWidget.h"

using namespace std;

struct SharedWidget::Impl
{
    int b;
    float f;
    int start;
};

SharedWidget::SharedWidget(int start_)
    : pImpl(make_shared<Impl>(0, 0.0f, start_))
{
}

int SharedWidget::add(int a, int b)
{
    return a+b+pImpl->start;
}
