#!/usr/bin/env python3
from setuptools import setup, Extension
ext_module=Extension("pygeometry", sources=["pygeom.cpp"], include_dirs=[pybind11.get_include()], language="c++20")
setup(name="pygeometry", ext_modules=[ext_module])