cmake_minimum_required(VERSION 3.4...3.18)
project(olcUTIL_Geometry2D_py)

#add_subdirectory(pybind11)
pybind11_add_module(${PROJECT_NAME} olcUTIL_Geometry2D_py.cpp)

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
# target_compile_definitions(${PROJECT_NAME} PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO)
