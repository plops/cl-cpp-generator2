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
    unique_ptr<IContainer> container;
};

SharedWidget::SharedWidget(int start_, unique_ptr<IContainer> container)
    : pImpl(make_shared<Impl>(0, 0.0f, start_, move(container)))
{
}

int SharedWidget::add(int a, int b)
{
    return a + b + pImpl->start;
}

void SharedWidget::insert(float f)
{
    pImpl->container->insert(f);
}
