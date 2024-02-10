python setup.py build_ext --inplace  # Build the module



# the following line fails like this on my system. i know this error can
# be avoided by creating a virtualenv, but i haven't looked into that.
#error: externally-managed-environment
#
#× This environment is externally managed
#╰─> 
#    The system-wide Python installation in Gentoo should be maintained
#    using the system package manager (e.g. emerge).
# 

# pip install .    


# for now i just manually copy the library from the build folder and run a test

cp build/lib.linux-x86_64-cpython-311/olcUTIL_Geometry2D_py.cpython-311-x86_64-linux-gnu.so .
python test_python.py


# the test fails:

# 144_geom2d/source01/python $ ./setup01_build.sh
# running build_ext
# INFO: Disabling color, you really want to install colorlog.
# Disabling color, you really want to install colorlog.
# copying build/lib.linux-x86_64-cpython-311/olcUTIL_Geometry2D_py.cpython-311-x86_64-linux-gnu.so -> 
# Traceback (most recent call last):
#   File "/home/martin/stage/cl-cpp-generator2/example/144_geom2d/source01/python/test_python.py", line 2, in <module>
#     from olcUTIL_Geometry2D_py import v, circle, rect, line, triangle, contains, closest, overlaps, intersects, envelope_c, envelope_r
# ImportError: /home/martin/stage/cl-cpp-generator2/example/144_geom2d/source01/python/olcUTIL_Geometry2D_py.cpython-311-x86_64-linux-gnu.so: undefined symbol: _ZNKSt13runtime_error4whatEv
