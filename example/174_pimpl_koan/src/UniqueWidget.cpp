//
// Created by martin on 4/5/25.
//

#include "UniqueWidget.h"
#include <memory>
using namespace std;


struct UniqueWidget::Impl
{
    int start;
    unique_ptr<IContainer> container;
};


UniqueWidget::UniqueWidget(int start, unique_ptr<IContainer> container)
    : pImpl{make_unique<Impl>(start, move(container))}
{
}

int UniqueWidget::add(int a, int b)
{
    return a + b + pImpl->start;
}

UniqueWidget::~UniqueWidget() = default; // destructor

UniqueWidget::UniqueWidget(const UniqueWidget& other) // copy ctor
    : IWidget{other}, pImpl{make_unique<Impl>(*other.pImpl)}
{
}

//UniqueWidget::UniqueWidget(UniqueWidget&& other) noexcept // move ctor
//: IWidget{move(other)},
//pImpl{move(other.pImpl)}
//{
//}

UniqueWidget::UniqueWidget(UniqueWidget&& rhs) noexcept = default; // move ctor
UniqueWidget& UniqueWidget::operator=(UniqueWidget&& rhs) noexcept = default; // move assign
void UniqueWidget::insert(float f)
{
    pImpl->container->insert(f);
}

UniqueWidget& UniqueWidget::operator=(const UniqueWidget& other) // copy assign
{
    if (this == &other)
        return *this;
    IWidget::operator =(other);
    *pImpl = *other.pImpl;
    return *this;
}

//
//UniqueWidget& UniqueWidget::operator=(UniqueWidget&& other) noexcept // move assign
//{
//    if (this == &other)
//        return *this;
//    IWidget::operator =(std::move(other));
//    pImpl = std::move(other.pImpl);
//    return *this;
//}
