//#include "shape.h"
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

class Circle
{
public:
    explicit Circle(double radius) : radius_{radius} {}

    [[nodiscard]] double radius() const { return radius_; }

private:
    double radius_;
};

class Shape
{
public:
    template <typename ConcreteShape, typename DrawStrategy>
    Shape(ConcreteShape shape, DrawStrategy drawer) :
        pimpl_{std::make_unique<ShapeModel<ConcreteShape, DrawStrategy>>(shape, drawer)}
    {
    }

    void draw() const { pimpl_->draw(); }
    // rule of 5, needs work
    Shape(Shape const& shape) : pimpl_{std::move(pimpl_)} {}

    Shape(Shape&& shape) noexcept : pimpl_{std::move(pimpl_)} {}

    ~Shape() = default;

    Shape& operator=(Shape const& shape)
    {
        pimpl_ = std::move(pimpl_);
        return *this;
    }

    Shape& operator=(Shape&& shape) noexcept
    {
        pimpl_ = std::move(pimpl_);
        return *this;
    }

private:
    class ShapeConcept
    {
    public:
        virtual ~ShapeConcept() = default;
        virtual void draw() const = 0;
        // virtual ShapeConcept *clone() const = 0; // is this the right clone()?
    };


    /** This class knows how to draw a shape */
    template <typename ConcreteShape, typename DrawStrategy>
    class ShapeModel : public ShapeConcept
    {
    public:
        explicit ShapeModel(ConcreteShape shape, DrawStrategy drawer) : shape_{shape}, drawer_{drawer} {}

        void draw() const override { drawer_(shape_); }
        // ShapeConcept *clone() const override { return new ShapeModel(shape_); }
    private:
        ConcreteShape shape_;
        DrawStrategy drawer_;
    };

    std::unique_ptr<ShapeConcept> pimpl_;
};

using Shapes = std::vector<Shape>;
using ShapesFactory = std::function<Shapes(std::string_view)>;

void drawAllShapes(Shapes const& shapes)
{
    for (auto const& shape : shapes)
    {
        shape.draw();
    }
}

void createAndDrawShapes(ShapesFactory const& factory, std::string_view filename)
{
    Shapes shapes = factory(filename);
    drawAllShapes(shapes);
}

class OpenGLDrawer
{
public:
    explicit OpenGLDrawer() = default;

    void operator()(Circle const& circle) const {};
};

class YourShapesFactory
{
public:
    Shapes operator()(std::string_view filename) const
    {
        Shapes shapes{};
        std::string shape{};
        std::istringstream shape_file(static_cast<std::string>(filename));
        while (shape_file >> shape)
        {
            if (shape == "circle")
            {
                double radius{};
                shape_file >> radius;
                shapes.emplace_back(Circle{radius}, OpenGLDrawer{});
            }
            else
            {
                break;
            }
        }
        return shapes;
    }
};

int main(int argc, const char* argv[])
{
    YourShapesFactory factory{};
    createAndDrawShapes(factory, "shapes.txt");
    return 0;
}
