#!/usr/bin/env python3
from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension
ext_modules=[Pybind11Extension("olcUTIL_Geometry2D_py", sources=["olcUTIL_Geometry2D_py.cpp"], language="c++20")]
setup(name="olcUTIL_Geometry2D_py", version="0.0.1", author="Developers of olcUTIL_Geometry2D", description="Python bindings for olcUTIL_Geometry2D library using Pybind11", ext_modules=ext_modules)