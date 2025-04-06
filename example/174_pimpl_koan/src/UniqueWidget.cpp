//
// Created by martin on 4/5/25.
//

#include "UniqueWidget.h"
#include <memory>
#include "Vec.h"
using namespace std;


struct UniqueWidget::Impl
{
    int start;
    unique_ptr<IContainer> container;
    ~Impl() = default;

    Impl(int start, unique_ptr<IContainer> container)
        : start(start),
          container(move(container))
    {
    }
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
    : //IWidget{other}
       pImpl{
          [&other]()
          {
              int n = other.pImpl->container->size();
              auto uv = make_unique<Vec>(n);
              for (int i=0;i<n;i++)
                  (*uv)[i]  = (*other.pImpl->container)[i];
              return make_unique<Impl>(other.pImpl->start, move(uv));
          }()
      }
{
}

//UniqueWidget::UniqueWidget(UniqueWidget&& other) noexcept // move ctor
//: IWidget{move(other)},
//pImpl{move(other.pImpl)}
//{
//}

UniqueWidget::UniqueWidget(UniqueWidget&& other) noexcept = default; // move ctor
UniqueWidget& UniqueWidget::operator=(UniqueWidget&& other) noexcept = default; // move assign
void UniqueWidget::insert(float f)
{
    pImpl->container->insert(f);
}

UniqueWidget& UniqueWidget::operator=(const UniqueWidget& other) // copy assign
{
    if (this == &other)
        return *this;
    pImpl->start = other.pImpl->start;
    *pImpl->container = *other.pImpl->container;
    return *this;
}

//UniqueWidget& UniqueWidget::operator=(UniqueWidget&& other) noexcept // move assign
//{
//    if (this == &other)
//        return *this;
//    IWidget::operator =(std::move(other));
//    pImpl = std::move(other.pImpl);
//    return *this;
//}
