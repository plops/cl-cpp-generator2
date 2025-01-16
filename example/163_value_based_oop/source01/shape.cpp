#include "shape.h"
#include <memory>
#include <vector>
#include <functional>
#include <string>

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
	// rule of 5, needs work
    Shape(Shape const& shape) : pimpl_{std::move(pimpl_)} {}
    Shape(Shape&& shape) : pimpl_{std::move(pimpl_)} {}
    ~Shape() = default;
    Shape& operator=(Shape const& shape) { pimpl_ = std::move(pimpl_); return *this; }
    Shape& operator=(Shape&& shape) { pimpl_ = std::move(pimpl_); return *this; }
  private:
class ShapeConcept
{
  public:
    virtual ~ShapeConcept() = default;
    virtual void draw() const = 0;
	virtual ShapeConcept *clone() const = 0; // is this the right clone()?
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

using Shapes = std::vector<Shape>;
using ShapesFactory = std::function<Shapes(std::string_view)>;

void drawAllShapes(Shapes const& shapes){
  for(auto const& shape : shapes){
    shape.draw();
  }
}

void createAndDrawShapes(ShapesFactory const& factory, std::string_view filename){
  Shapes shapes = factory(filename);
  drawAllShapes(shapes);
}

