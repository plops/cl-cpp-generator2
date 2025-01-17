#include <functional>
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
