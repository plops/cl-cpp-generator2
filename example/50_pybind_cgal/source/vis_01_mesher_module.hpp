#ifndef VIS_01_MESHER_MODULE_H
#define VIS_01_MESHER_MODULE_H
#include "utils.h"
;
#include "globals.h"
;
#include <chrono>
#include <iostream>
#include <thread>
;
#include <cxxabi.h>
;
#include <pybind11/pybind11.h>
;
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
;
#include <CGAL/Delaunay_mesh_face_base_2.h>
;
#include <CGAL/Delaunay_mesh_vertex_base_2.h>
;
#include <CGAL/Constrained_Delaunay_triangulation_2.h>
;
#include <CGAL/Delaunay_mesh_size_criteria_2.h>
;
#include <CGAL/Delaunay_mesher_2.h>
;
#include <CGAL/Triangulation_conformer_2.h>
;
#include <CGAL/lloyd_optimize_mesh_2.h>
;
// header
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <pybind11/pybind11.h>

#include <CGAL/Delaunay_mesh_face_base_2.h>

#include <CGAL/Delaunay_mesh_vertex_base_2.h>

#include <CGAL/Constrained_Delaunay_triangulation_2.h>

#include <CGAL/Delaunay_mesh_size_criteria_2.h>

#include <CGAL/Delaunay_mesher_2.h>

#include <CGAL/Triangulation_conformer_2.h>

#include <CGAL/lloyd_optimize_mesh_2.h>
;
std::string demangle(const std::string name);
template <class T> std::string type_name();
#endif