//
// Created by martin on 1/17/25.
//

#ifndef SHAPE_H
#define SHAPE_H

#include <memory>

class Shape
{
public:
    template <typename ConcreteShape, typename DrawStrategy>
    Shape(ConcreteShape shape, DrawStrategy drawer) :
        pimpl_{std::make_unique<ShapeModel<ConcreteShape, DrawStrategy>>(shape, drawer)}
    {
    }

    void draw() const;
    // rule of 5, needs work
    Shape(Shape const& shape);

    Shape(Shape&& shape) noexcept;

    ~Shape() = default;

    Shape& operator=(Shape const& shape);

    Shape& operator=(Shape&& shape) noexcept;

private:
    class ShapeConcept
    {
    public:
        virtual ~ShapeConcept() = default;
        virtual void draw() const = 0;
        // cpp design guideline 32

        virtual std::unique_ptr<ShapeConcept> clone() const = 0;
    };


    /** This class knows how to draw a shape */
    template <typename ConcreteShape, typename DrawStrategy>
    class ShapeModel : public ShapeConcept
    {
    public:
        explicit ShapeModel(ConcreteShape shape, DrawStrategy drawer) : shape_{shape}, drawer_{drawer} {}

        void draw() const override { drawer_(shape_); }
        std::unique_ptr<ShapeConcept> clone() const override
        {
            return std::make_unique<ShapeModel>(*this);
        }
    private:
        ConcreteShape shape_;
        DrawStrategy drawer_;
    };

    std::unique_ptr<ShapeConcept> pimpl_;
};

#endif // SHAPE_H
