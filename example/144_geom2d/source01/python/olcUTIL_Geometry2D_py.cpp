// get header with: wget
// https://raw.githubusercontent.com/OneLoneCoder/olcUTIL_Geometry2D/main/olcUTIL_Geometry2D.h

// compile with: c++ -O3 -Wall -shared -std=c++20 -fPIC $(python3 -m pybind11
// --includes) olcUTIL_Geometry2D_py.cpp -o
// olcUTIL_Geometry2D_py$(python3-config --extension-suffix)

#include "olcUTIL_Geometry2D.h"
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;
using namespace olc::utils::geom2d;
using namespace olc;
PYBIND11_MODULE(olcUTIL_Geometry2D_py, m) {
  // Expose the v_2d<float> class to Python as "Vector2D"

  py::class_<v_2d<float>>(m, "Vector2D")
      .def(py::init<float, float>())
      .def_readwrite("x", &v_2d<float>::x)
      .def_readwrite("y", &v_2d<float>::y)
      .def("__repr__",
           [](const v_2d<float> &v) {
             return "<Vector2D x=" + std::to_string(v.x) +
                    ", y=" + std::to_string(v.y) + ">";
           })
      .def("area", &v_2d<float>::area)
      .def("mag", &v_2d<float>::mag)
      .def("mag2", &v_2d<float>::mag2)
      .def("norm", &v_2d<float>::norm)
      .def("perp", &v_2d<float>::perp)
      .def("floor", &v_2d<float>::floor)
      .def("ceil", &v_2d<float>::ceil)
      .def("max", &v_2d<float>::max)
      .def("min", &v_2d<float>::min)
      .def("dot", &v_2d<float>::dot)
      .def("cross", &v_2d<float>::cross)
      .def("cart", &v_2d<float>::cart)
      .def("polar", &v_2d<float>::polar)
      .def("clamp", &v_2d<float>::clamp)
      .def("lerp", &v_2d<float>::lerp)
      .def("reflect", &v_2d<float>::reflect)
      .def("__str__", &v_2d<float>::str);
  m.attr("pi") = utils::geom2d::pi;
  m.attr("epsilon") = utils::geom2d::epsilon;
  // Expose the line<float> class to Python as "Line"

  py::class_<line<float>>(m, "Line")
      .def(py::init<const v_2d<float> &, const v_2d<float> &>())
      .def_readwrite("start", &line<float>::start)
      .def_readwrite("end", &line<float>::end);
  // Expose the circle<float> class to Python as "Circle"

  py::class_<circle<float>>(m, "Circle")
      .def(py::init<const v_2d<float> &, float>())
      .def_readwrite("pos", &circle<float>::pos)
      .def_readwrite("radius", &circle<float>::radius);
  // Expose the contains function for circle and point

  m.def("contains",
        (bool (*)(const circle<float> &, const v_2d<float> &)) & contains);
};