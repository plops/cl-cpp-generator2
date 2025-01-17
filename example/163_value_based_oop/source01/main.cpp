#include <functional>
#include <iostream>
#include <ostream>
#include <sstream>
#include <string_view>
#include <vector>
#include "Circle.h"
#include "Shape.h"


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

    void operator()(Circle const& circle) const
    {
        std::cout << std::format("draw circle {}",circle.radius()) << std::endl;
    };
};

class YourShapesFactory
{
public:
    Shapes operator()(std::string_view filename) const
    {
        std::cout << std::format("YourShapesFactory {}", filename) << std::endl;
        Shapes shapes{};
        std::string shape{};
        std::istringstream shape_file(static_cast<std::string>(filename));
        while (shape_file >> shape)
        {
            if (shape == "circle")
            {
                double radius{};
                shape_file >> radius;
                std::cout << std::format("  Circle {}", radius) << std::endl;

                shapes.emplace_back(Circle{radius}, OpenGLDrawer{});
            }
            else
            {
                std::cout << std::format("  unsupported Shape: {}", shape) << std::endl;
                break;
            }
        }
        return shapes;
    }
};

int main(int argc, const char* argv[])
{
    YourShapesFactory factory{};
    createAndDrawShapes(factory, "circle 5");
    return 0;
}
