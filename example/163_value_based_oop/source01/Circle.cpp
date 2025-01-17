//
// Created by martin on 1/17/25.
//

#include "Circle.h"

Circle::Circle(double radius) : radius_{radius} {}

[[nodiscard]] double Circle::radius() const { return radius_; }
