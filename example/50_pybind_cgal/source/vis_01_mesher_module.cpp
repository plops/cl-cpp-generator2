
#include "utils.h"

#include "globals.h"

extern State state;
#include <chrono>
#include <iostream>
#include <thread>

// implementation
#include "vis_01_mesher_module.hpp"
using K = CGAL::Exact_predicates_inexact_constructions_kernel;
using Fb = CGAL::Delaunay_mesh_face_base_2<K>;
using Vb = CGAL::Delaunay_mesh_vertex_base_2<K>;
using Tds = CGAL::Triangulation_data_structure_2<Vb, Fb>;
using CDT = CGAL::Constrained_Delaunay_triangulation_2<K, Tds>;
using Criteria = CGAL::Delaunay_mesh_size_criteria_2<CDT>;
using Mesher = CGAL::Delaunay_mesher_2<CDT, Criteria>;
using Vertex_handle = CDT::Vertex_handle;
using Point = CDT::Point;
namespace py = pybind11;
using namespace std::chrono_literals;

PYBIND11_MODULE(cgal_mesher, m) {
  py::class_<Point>(m, "Point")
      .def(py::init<int, int>(), py::arg("x"), py::arg("y"))
      .def(py::init<double, double>(), py::arg("x"), py::arg("y"))
      .def_property_readonly("x", &Point::x)
      .def_property_readonly("y", &Point::y)
      .def("__repr__", [](const Point &p) {
        auto r = std::string("Point(");
        (r) += (std::to_string(p.x()));
        (r) += (", ");
        (r) += (std::to_string(p.x()));
        (r) += (")");
        return r;
      });
  py::class_<Vertex_handle>(m, "VertexHandle");
  py::class_<CDT>(m, "ConstrainedDelaunayTriangulation")
      .def(py::init())
      .def("insert", [](CDT &cdt, const Point &p) { return cdt.insert(p); })
      .def("insert_constraint",
           [](CDT &cdt, Vertex_handle a, Vertex_handle b) {
             return cdt.insert_constraint(a, b);
           })
      .def("number_of_vertices", &CDT::number_of_vertices)
      .def("number_of_faces", &CDT::number_of_faces);
};