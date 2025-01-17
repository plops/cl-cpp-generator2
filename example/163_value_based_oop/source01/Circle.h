//
// Created by martin on 1/17/25.
//

#ifndef CIRCLE_H
#define CIRCLE_H


class Circle
{
public:
    explicit Circle(double radius);

    [[nodiscard]] double radius() const;

private:
    double radius_;
};


#endif //CIRCLE_H
