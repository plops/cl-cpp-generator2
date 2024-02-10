// get header with: wget
// https://raw.githubusercontent.com/OneLoneCoder/olcUTIL_Geometry2D/main/olcUTIL_Geometry2D.h

// compile with: c++ -O3 -Wall -shared -std=c++20 -fPIC $(python3 -m pybind11
// --includes) olcUTIL_Geometry2D_py.cpp -o
// olcUTIL_Geometry2D_py$(python3-config --extension-suffix)

#include "olcUTIL_Geometry2D.h"
#include <pybind11/pybind11.h>
namespace py = pybind11;
using namespace olc::utils::geom2d;
using namespace olc;
PYBIND11_MODULE(olcUTIL_Geometry2D_py, m) {
  // Expose the v_2d<float> class to Python as "v"

  py::class_<v_2d<float>>(m, "v")
      .def(py::init<float, float>())
      .def_readwrite("x", &v_2d<float>::x)
      .def_readwrite("y", &v_2d<float>::y)
      .def("__repr__",
           [](const v_2d<float> &v) {
             return "<v x=" + std::to_string(v.x) +
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
  // Expose the line<float> class to Python as "line"

  py::class_<line<float>>(m, "line")
      .def(py::init<const v_2d<float> &, const v_2d<float> &>())
      .def_readwrite("start", &line<float>::start)
      .def_readwrite("end", &line<float>::end)
      .def("__repr__",
           [](const line<float> &arg) {
             return std::string("<line") + " start=" + arg.start.str() +
                    " end=" + arg.end.str() + ">";
           })
      .def("length", &line<float>::length)
      .def("vector", &line<float>::vector);
  // Expose the rect<float> class to Python as "rect"

  py::class_<rect<float>>(m, "rect")
      .def(py::init<const v_2d<float> &, const v_2d<float> &>())
      .def_readwrite("pos", &rect<float>::pos)
      .def_readwrite("size", &rect<float>::size)
      .def("__repr__",
           [](const rect<float> &arg) {
             return std::string("<rect") + " pos=" + arg.pos.str() +
                    " size=" + arg.size.str() + ">";
           })
      .def("area", &rect<float>::area);
  // Expose the triangle<float> class to Python as "triangle"

  py::class_<triangle<float>>(m, "triangle")
      .def(py::init<const v_2d<float> &, const v_2d<float> &,
                    const v_2d<float> &>())
      .def("__repr__",
           [](const triangle<float> &arg) {
             return std::string("<triangle") + ">";
           })
      .def("area", &triangle<float>::area);
  // Expose the circle<float> class to Python as "circle"

  py::class_<circle<float>>(m, "circle")
      .def(py::init<const v_2d<float> &, float>())
      .def_readwrite("pos", &circle<float>::pos)
      .def_readwrite("radius", &circle<float>::radius)
      .def("__repr__",
           [](const circle<float> &arg) {
             return std::string("<circle") + " pos=" + arg.pos.str() +
                    " radius=" + std::to_string(arg.radius) + ">";
           })
      .def("area", &circle<float>::area);
  m.def("envelope_c", [](const v_2d<float> &arg) { return envelope_c(arg); });
  m.def("envelope_r", [](const v_2d<float> &arg) { return envelope_r(arg); });
  m.def("envelope_c", [](const circle<float> &arg) { return envelope_c(arg); });
  m.def("envelope_r", [](const circle<float> &arg) { return envelope_r(arg); });
  m.def("envelope_c", [](const line<float> &arg) { return envelope_c(arg); });
  m.def("envelope_r", [](const line<float> &arg) { return envelope_r(arg); });
  m.def("envelope_c", [](const rect<float> &arg) { return envelope_c(arg); });
  m.def("envelope_r", [](const rect<float> &arg) { return envelope_r(arg); });
  m.def("envelope_c",
        [](const triangle<float> &arg) { return envelope_c(arg); });
  m.def("envelope_r",
        [](const triangle<float> &arg) { return envelope_r(arg); });
  // contains(triangle,rect)

  m.def("contains",
        (bool (*)(const triangle<float> &, const rect<float> &)) & contains);
  // overlaps(triangle,rect)

  m.def("overlaps",
        (bool (*)(const triangle<float> &, const rect<float> &)) & overlaps);
  // contains(rect,triangle)

  m.def("contains",
        (bool (*)(const rect<float> &, const triangle<float> &)) & contains);
  // overlaps(rect,triangle)

  m.def("overlaps",
        (bool (*)(const rect<float> &, const triangle<float> &)) & overlaps);
  // contains(triangle,line)

  m.def("contains",
        (bool (*)(const triangle<float> &, const line<float> &)) & contains);
  // overlaps(triangle,line)

  m.def("overlaps",
        (bool (*)(const triangle<float> &, const line<float> &)) & overlaps);
  // contains(line,triangle)

  m.def("contains",
        (bool (*)(const line<float> &, const triangle<float> &)) & contains);
  // overlaps(line,triangle)

  m.def("overlaps",
        (bool (*)(const line<float> &, const triangle<float> &)) & overlaps);
  // contains(rect,line)

  m.def("contains",
        (bool (*)(const rect<float> &, const line<float> &)) & contains);
  // overlaps(rect,line)

  m.def("overlaps",
        (bool (*)(const rect<float> &, const line<float> &)) & overlaps);
  // contains(line,rect)

  m.def("contains",
        (bool (*)(const line<float> &, const rect<float> &)) & contains);
  // overlaps(line,rect)

  m.def("overlaps",
        (bool (*)(const line<float> &, const rect<float> &)) & overlaps);
  // contains(triangle,circle)

  m.def("contains",
        (bool (*)(const triangle<float> &, const circle<float> &)) & contains);
  // overlaps(triangle,circle)

  m.def("overlaps",
        (bool (*)(const triangle<float> &, const circle<float> &)) & overlaps);
  // contains(circle,triangle)

  m.def("contains",
        (bool (*)(const circle<float> &, const triangle<float> &)) & contains);
  // overlaps(circle,triangle)

  m.def("overlaps",
        (bool (*)(const circle<float> &, const triangle<float> &)) & overlaps);
  // contains(rect,circle)

  m.def("contains",
        (bool (*)(const rect<float> &, const circle<float> &)) & contains);
  // overlaps(rect,circle)

  m.def("overlaps",
        (bool (*)(const rect<float> &, const circle<float> &)) & overlaps);
  // contains(circle,rect)

  m.def("contains",
        (bool (*)(const circle<float> &, const rect<float> &)) & contains);
  // overlaps(circle,rect)

  m.def("overlaps",
        (bool (*)(const circle<float> &, const rect<float> &)) & overlaps);
  // contains(line,circle)

  m.def("contains",
        (bool (*)(const line<float> &, const circle<float> &)) & contains);
  // overlaps(line,circle)

  m.def("overlaps",
        (bool (*)(const line<float> &, const circle<float> &)) & overlaps);
  // contains(circle,line)

  m.def("contains",
        (bool (*)(const circle<float> &, const line<float> &)) & contains);
  // overlaps(circle,line)

  m.def("overlaps",
        (bool (*)(const circle<float> &, const line<float> &)) & overlaps);
  // contains(triangle,v_2d)

  m.def("contains",
        (bool (*)(const triangle<float> &, const v_2d<float> &)) & contains);
  // overlaps(triangle,v_2d)

  m.def("overlaps",
        (bool (*)(const triangle<float> &, const v_2d<float> &)) & overlaps);
  // contains(v_2d,triangle)

  m.def("contains",
        (bool (*)(const v_2d<float> &, const triangle<float> &)) & contains);
  // overlaps(v_2d,triangle)

  m.def("overlaps",
        (bool (*)(const v_2d<float> &, const triangle<float> &)) & overlaps);
  // contains(rect,v_2d)

  m.def("contains",
        (bool (*)(const rect<float> &, const v_2d<float> &)) & contains);
  // overlaps(rect,v_2d)

  m.def("overlaps",
        (bool (*)(const rect<float> &, const v_2d<float> &)) & overlaps);
  // contains(v_2d,rect)

  m.def("contains",
        (bool (*)(const v_2d<float> &, const rect<float> &)) & contains);
  // overlaps(v_2d,rect)

  m.def("overlaps",
        (bool (*)(const v_2d<float> &, const rect<float> &)) & overlaps);
  // contains(line,v_2d)

  m.def("contains",
        (bool (*)(const line<float> &, const v_2d<float> &)) & contains);
  // overlaps(line,v_2d)

  m.def("overlaps",
        (bool (*)(const line<float> &, const v_2d<float> &)) & overlaps);
  // contains(v_2d,line)

  m.def("contains",
        (bool (*)(const v_2d<float> &, const line<float> &)) & contains);
  // overlaps(v_2d,line)

  m.def("overlaps",
        (bool (*)(const v_2d<float> &, const line<float> &)) & overlaps);
  // contains(circle,v_2d)

  m.def("contains",
        (bool (*)(const circle<float> &, const v_2d<float> &)) & contains);
  // overlaps(circle,v_2d)

  m.def("overlaps",
        (bool (*)(const circle<float> &, const v_2d<float> &)) & overlaps);
  // contains(v_2d,circle)

  m.def("contains",
        (bool (*)(const v_2d<float> &, const circle<float> &)) & contains);
  // overlaps(v_2d,circle)

  m.def("overlaps",
        (bool (*)(const v_2d<float> &, const circle<float> &)) & overlaps);
  // intersects(triangle,rect)

  m.def("intersects", [](const triangle<float> &arg1, const rect<float> &arg2) {
    return intersects(arg1, arg2);
  });
  // intersects(rect,triangle)

  m.def("intersects", [](const rect<float> &arg1, const triangle<float> &arg2) {
    return intersects(arg1, arg2);
  });
  // intersects(triangle,line)

  m.def("intersects", [](const triangle<float> &arg1, const line<float> &arg2) {
    return intersects(arg1, arg2);
  });
  // intersects(line,triangle)

  m.def("intersects", [](const line<float> &arg1, const triangle<float> &arg2) {
    return intersects(arg1, arg2);
  });
  // intersects(rect,line)

  m.def("intersects", [](const rect<float> &arg1, const line<float> &arg2) {
    return intersects(arg1, arg2);
  });
  // intersects(line,rect)

  m.def("intersects", [](const line<float> &arg1, const rect<float> &arg2) {
    return intersects(arg1, arg2);
  });
  // intersects(triangle,circle)

  m.def("intersects",
        [](const triangle<float> &arg1, const circle<float> &arg2) {
          return intersects(arg1, arg2);
        });
  // intersects(circle,triangle)

  m.def("intersects",
        [](const circle<float> &arg1, const triangle<float> &arg2) {
          return intersects(arg1, arg2);
        });
  // intersects(rect,circle)

  m.def("intersects", [](const rect<float> &arg1, const circle<float> &arg2) {
    return intersects(arg1, arg2);
  });
  // intersects(circle,rect)

  m.def("intersects", [](const circle<float> &arg1, const rect<float> &arg2) {
    return intersects(arg1, arg2);
  });
  // intersects(line,circle)

  m.def("intersects", [](const line<float> &arg1, const circle<float> &arg2) {
    return intersects(arg1, arg2);
  });
  // intersects(circle,line)

  m.def("intersects", [](const circle<float> &arg1, const line<float> &arg2) {
    return intersects(arg1, arg2);
  });
  // intersects(triangle,v_2d)

  m.def("intersects", [](const triangle<float> &arg1, const v_2d<float> &arg2) {
    return intersects(arg1, arg2);
  });
  // intersects(v_2d,triangle)

  m.def("intersects", [](const v_2d<float> &arg1, const triangle<float> &arg2) {
    return intersects(arg1, arg2);
  });
  // intersects(rect,v_2d)

  m.def("intersects", [](const rect<float> &arg1, const v_2d<float> &arg2) {
    return intersects(arg1, arg2);
  });
  // intersects(v_2d,rect)

  m.def("intersects", [](const v_2d<float> &arg1, const rect<float> &arg2) {
    return intersects(arg1, arg2);
  });
  // intersects(line,v_2d)

  m.def("intersects", [](const line<float> &arg1, const v_2d<float> &arg2) {
    return intersects(arg1, arg2);
  });
  // intersects(v_2d,line)

  m.def("intersects", [](const v_2d<float> &arg1, const line<float> &arg2) {
    return intersects(arg1, arg2);
  });
  // intersects(circle,v_2d)

  m.def("intersects", [](const circle<float> &arg1, const v_2d<float> &arg2) {
    return intersects(arg1, arg2);
  });
  // intersects(v_2d,circle)

  m.def("intersects", [](const v_2d<float> &arg1, const circle<float> &arg2) {
    return intersects(arg1, arg2);
  });
  // closest(v_2d,v_2d)

  m.def("closest", [](const v_2d<float> &arg1, const v_2d<float> &arg2) {
    return closest(arg1, arg2);
  });
  // closest(v_2d,line)

  m.def("closest", [](const v_2d<float> &arg1, const line<float> &arg2) {
    return closest(arg1, arg2);
  });
  // closest(v_2d,rect)

  m.def("closest", [](const v_2d<float> &arg1, const rect<float> &arg2) {
    return closest(arg1, arg2);
  });
  // closest(v_2d,circle)

  m.def("closest", [](const v_2d<float> &arg1, const circle<float> &arg2) {
    return closest(arg1, arg2);
  });
  // closest(v_2d,triangle)

  m.def("closest", [](const v_2d<float> &arg1, const triangle<float> &arg2) {
    return closest(arg1, arg2);
  });
  // closest(line,v_2d)

  m.def("closest", [](const line<float> &arg1, const v_2d<float> &arg2) {
    return closest(arg1, arg2);
  });
  // closest(line,circle)

  m.def("closest", [](const line<float> &arg1, const circle<float> &arg2) {
    return closest(arg1, arg2);
  });
  // closest(rect,v_2d)

  m.def("closest", [](const rect<float> &arg1, const v_2d<float> &arg2) {
    return closest(arg1, arg2);
  });
  // closest(circle,v_2d)

  m.def("closest", [](const circle<float> &arg1, const v_2d<float> &arg2) {
    return closest(arg1, arg2);
  });
  // closest(circle,line)

  m.def("closest", [](const circle<float> &arg1, const line<float> &arg2) {
    return closest(arg1, arg2);
  });
  // closest(triangle,v_2d)

  m.def("closest", [](const triangle<float> &arg1, const v_2d<float> &arg2) {
    return closest(arg1, arg2);
  });
};