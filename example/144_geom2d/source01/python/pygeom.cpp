#include "olcUTIL_Geometry2D.h"
#include <pybind11/pybind11.h>
namespace py = pybind11;
using namespace olc;
PYBIND11_MODULE(pygeometry, m) {
  py::class_<v_2d<float>>(m, "v_2d")
      .def(py::init<float, float>())
      .def_readwrite("x", &v_2d<float>::x)
      .def_readwrite("y", &v_2d<float>::y)
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
      .def(py::self, ==, py::self)
      .def(py::self, !=, py::self)
      .def(py::self, +, py::self)
      .def(py::self, -, py::self)
      .def(py::self, *, py::self)
      .def(py::self, /, py::self)
      .def(py::self, +=, py::self)
      .def(py::self, -=, py::self)
      .def(py::self, *=, float())
      .def(py::self, /=, float())
      .def("__str__", &v_2d<float>::str)
      .def("__repr__", &v_2d<float>::str);
  m.def([](float lhs, v_2d<float> &rhs) { return lhs * rhs; });
  m.def([](v_2d<float> &lhs, float rhs) { return lhs * rhs; });
  m.def([](float lhs, v_2d<float> &rhs) { return lhs + rhs; });
  m.def([](v_2d<float> &lhs, float rhs) { return lhs + rhs; });
  m.attr("pi") = utils::geom2d::pi;
  m.attr("epsilon") = utils::geom2d::epsilon;
  py::class_<utils::geom2d::line<float>>(m, "line")
      .def(py::init<v_2d<float>, v_2d<float>>())
      .def("vector", &utils::geom2d::line<float>::vector)
      .def("length", &utils::geom2d::line<float>::length)
      .def("length2", &utils::geom2d::line<float>::length2)
      .def("rpoint", &utils::geom2d::line<float>::rpoint)
      .def("upoint", &utils::geom2d::line<float>::upoint)
      .def("side", &utils::geom2d::line<float>::side)
      .def("coefficients", &utils::geom2d::line<float>::coefficients);
};