//
// Created by martin on 1/16/25.
//

#include "shape.h"


class ShapeConcept
{
  public:
    virtual ~ShapeConcept() = default;
    virtual void draw() const = 0;
	virtual ShapeConcept *clone() const = 0;
};

template< typename ConcreteShape, typename DrawStrategy >
class ShapeModel : public ShapeConcept
{
  public:
    explicit ShapeModel(ConcreteShape shape, DrawStrategy drawer)
        : shape_(shape), drawer_(drawer)
    {}
    void draw() const override {drawer_(shape_);}
    ShapeConcept *clone() const override { return new ShapeModel(m_shape); }
  private:
    ConcreteShape shape_;
    DrawStrategy drawer_;
};

