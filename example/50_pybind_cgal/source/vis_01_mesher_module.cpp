
#include "utils.h"

#include "globals.h"

#include <chrono>
#include <iostream>
#include <thread>

#include <cxxabi.h>

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

template <typename T> class TypedInputIterator {
public:
  using iterator_category = std::input_iterator_tag;
  using difference_type = std::ptrdiff_t;
  using value_type = T;
  using pointer = T *;
  using reference = T &;
  explicit TypedInputIterator(py::iterator &py_iter) : py_iter_(py_iter) {}
  explicit TypedInputIterator(py::iterator &&py_iter) : py_iter_(py_iter) {}
  value_type operator*() { return (*py_iter_).template cast<value_type>(); }
  TypedInputIterator operator++(int inc) {
    auto copy = *this;
    ++py_iter_;
    return copy;
  }
  TypedInputIterator &operator++() {
    ++py_iter_;
    return *this;
  }
  bool operator!=(TypedInputIterator &rhs) {
    return (py_iter_) != (rhs.py_iter_);
  }
  bool operator==(TypedInputIterator &rhs) {
    return (py_iter_) == (rhs.py_iter_);
  }

private:
  py::iterator py_iter_;
};
std::string demangle(const std::string name) {
  auto status = -4;
  std::unique_ptr<char, void (*)(void *)> res{
      abi::__cxa_demangle(name.c_str(), nullptr, nullptr, &status), std::free};
  if ((0) == (status)) {
    return res.get();
  } else {
    return name;
  }
}
template <class T> std::string type_name() {
  typedef typename std::remove_reference<T>::type TR;
  std::unique_ptr<char, void (*)(void *)> own(nullptr, std::free);
  std::string r = (own != nullptr) ? own.get() : typeid(TR).name();
  r = demangle(r);
  if (std::is_const<TR>::value) {
    (r) += (" const");
  }
  if (std::is_volatile<TR>::value) {
    (r) += (" volatile");
  }
  if (std::is_lvalue_reference<TR>::value) {
    (r) += ("&");
  }
  if (std::is_rvalue_reference<TR>::value) {
    (r) += ("&&");
  }
  return r;
}
PYBIND11_MODULE(cgal_mesher, m) {
  py::class_<Point>(m, "Point")
      .def(py::init<int, int>(), py::arg("x"), py::arg("y"))
      .def(py::init<double, double>(), py::arg("x"), py::arg("y"))
      .def_property_readonly("x", &Point::x)
      .def_property_readonly("y", &Point::y)
      .def("__repr__",
           [](const Point &p) {
             auto r = std::string("Point(");
             (r) += (std::to_string(p.x()));
             (r) += (", ");
             (r) += (std::to_string(p.x()));
             (r) += (")");
             return r;
           })
      .def("__eq__", [](const Point &p, const Point &q) { return (p) == (q); })
      .def("__hash__", [](const Point &p) {
        std::hash<double> dh;
        auto xh = dh(p.x());
        auto yh = dh(p.y());
        return ((((yh) ^ (xh))) + (0x9E3779B9) + ((yh) << (6)) + ((yh) >> (2)));
      });
  py::class_<Vertex_handle>(m, "VertexHandle");
  m.def("print_faces_iterator_value_type", []() {
    (std::cout) << (type_name<CDT::Finite_faces_iterator::value_type>())
                << (std::endl);
  });
  py::class_<CDT>(m, "ConstrainedDelaunayTriangulation")
      .def(py::init())
      .def("insert", [](CDT &cdt, const Point &p) { return cdt.insert(p); })
      .def("insert_constraint",
           [](CDT &cdt, Vertex_handle a, Vertex_handle b) {
             return cdt.insert_constraint(a, b);
           })
      .def("number_of_vertices", &CDT::number_of_vertices)
      .def("number_of_faces", &CDT::number_of_faces)
      .def("finite_vertices",
           [](CDT &cdt) -> py::iterator {
             return py::make_iterator(cdt.finite_vertices_begin(),
                                      cdt.finite_vertices_end());
           })
      .def("finite_faces", [](CDT &cdt) -> py::iterator {
        return py::make_iterator(cdt.finite_faces_begin(),
                                 cdt.finite_faces_end());
      });
  py::class_<Criteria>(m, "Criteria")
      .def(py::init<double, double>(), py::arg("aspect_bound") = 0.125,
           py::arg("size_bound") = 0.0)
      .def(py::init<float, float>(), py::arg("aspect_bound") = 0.125,
           py::arg("size_bound") = 0.0)
      .def_property("size_bound", &Criteria::size_bound,
                    &Criteria::set_size_bound)
      .def_property(
          "aspect_bound", [](const Criteria &c) { return c.bound(); },
          [](Criteria &c, double bound) { c.set_bound(bound); });
  py::class_<Mesher>(m, "Mesher")
      .def(py::init<CDT &>())
      .def("seeds_from",
           [](Mesher &mesher, py::iterable iterable) {
             // cast the iterator to Point, otherwise it would assume python
             // object type
             ;
             auto it = py::iter(iterable);
             auto beg = TypedInputIterator<Point>(it);
             auto end = TypedInputIterator<Point>(py::iterator::sentinel());
             mesher.set_seeds(beg, end);
           })
      .def("refine_mesh", &Mesher::refine_mesh)
      .def_property("criteria", &Mesher::get_criteria,
                    [](Mesher &mesher, const Criteria &criteria) {
                      mesher.set_criteria(criteria);
                    });
  m.def("make_conforming_delaunay", &CGAL::make_conforming_Delaunay_2<CDT>,
        py::arg("cdt"),
        "Make a triangulation conform to the Delaunay property");
  m.def("make_conforming_gabriel", &CGAL::make_conforming_Gabriel_2<CDT>,
        py::arg("cdt"), "Make a triangulation conform to the Gabriel property");
  // boost named argument magic
  ;
  m.def(
      "lloyd_optimize",
      [](CDT &cdt, int max_iteration_number, double time_limit,
         double convergence, double freeze_bound) {
        CGAL::lloyd_optimize_mesh_2(
            cdt, CGAL::parameters::max_iteration_number = max_iteration_number,
            CGAL::parameters::time_limit = time_limit,
            CGAL::parameters::convergence = convergence,
            CGAL::parameters::freeze_bound = freeze_bound);
      },
      py::arg("cdt"), py::arg("max_iteration_number") = 0,
      py::arg("time_limit") = (0.), py::arg("convergence") = (1.00e-3),
      py::arg("freeze_bound") = (1.00e-3));
};