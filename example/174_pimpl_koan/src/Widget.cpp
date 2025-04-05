//
// Created by martin on 4/5/25.
//

#include "Widget.h"

using namespace std;

struct Widget::Impl
{
    int b;
    float f;
    int start;
};

Widget::Widget(int start_)
    : pImpl(make_shared<Impl>(0, 0.0f, start_))
{
}

int Widget::add(int a, int b)
{
    return a+b+pImpl->start;
}
