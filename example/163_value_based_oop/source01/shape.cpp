//
// Created by martin on 1/16/25.
//

#include "shape.h"
// we need unique_ptr
#include <memory>

class Circle {
  public:
    explicit Circle(double radius) : radius_{radius} {}
    double radius() const { return radius_; }
  private:
    double radius_;
};

class Shape {
  public:
    template< typename ConcreteShape, typename DrawStrategy >
    Shape(ConcreteShape shape, DrawStrategy drawer)
      : pimpl_{std::make_unique<ShapeModel<ConcreteShape, DrawStrategy>>(shape, drawer)}
    {}
    void draw() const { pimpl_->draw(); }

  private:
class ShapeConcept
{
  public:
    virtual ~ShapeConcept() = default;
    virtual void draw() const = 0;
	virtual ShapeConcept *clone() const = 0;
};


/** This class knows how to draw a shape */
template< typename ConcreteShape, typename DrawStrategy >
class ShapeModel : public ShapeConcept
{
  public:
    explicit ShapeModel(ConcreteShape shape, DrawStrategy drawer)
        : shape_{shape}, drawer_{drawer}
    {}
    void draw() const override {drawer_(shape_);}
    ShapeConcept *clone() const override { return new ShapeModel(shape_); }
  private:
    ConcreteShape shape_;
    DrawStrategy drawer_;
};
  std::unique_ptr<ShapeConcept> pimpl_;
};
