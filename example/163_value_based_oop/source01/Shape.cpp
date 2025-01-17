//
// Created by martin on 1/17/25.
//

#include "Shape.h"

void Shape::draw() const { pimpl_->draw(); }

Shape::Shape(Shape const& shape) : pimpl_{std::move(pimpl_)} {}

Shape::Shape(Shape&& shape) noexcept : pimpl_{std::move(pimpl_)} {}


Shape& Shape::operator=(Shape const& shape)
{
    pimpl_ = std::move(pimpl_);
    return *this;
}

Shape& Shape::operator=(Shape&& shape) noexcept
{
    pimpl_ = std::move(pimpl_);
    return *this;
}