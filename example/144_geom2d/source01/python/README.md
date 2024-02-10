**olcUTIL_Geometry2D: Python Bindings for 2D Geometry Calculations**

**About**

This project provides easy-to-use Python bindings for the 2D geometry
functions defined in the [olcUTIL_Geometry2D.h]
([https://github.com/OneLoneCoder/olcUTIL_Geometry2D/blob/main/olcUTIL_Geometry2D.h](https://github.com/OneLoneCoder/olcUTIL_Geometry2D/blob/main/olcUTIL_Geometry2D.h))
C++ header. Simplify your geometric calculations within Python using
these bindings.

**Features**

* **Core Functionality:** Currently exposes the `contains`, `overlaps`, `intersects`, `closest`, `envelope_r` and `envelope_b` function for all combinations of the following geometric types: 
    *  `v` (formerly `v_2d`): Represents a 2D vector/point.
    * `circle` 
    * `rect` (rectangle)
    * `line`
    * `triangle`
* **Planned Expansion:** Support for the `ray` object and the following functions from olcUTIL_Geometry2D is intended:
    * `reflects`
    * `collision`
	

**Building the Module**

1. **Auto-generated C++ Code:** The C++ binding code is automatically
   generated, enabling straightforward creation of instances for all
   permutations.

2. **Compiler Command:**
   ```
   c++ -O3 -Wall -shared -std=c++20 -fPIC $(python3 -m pybind11 --includes) \
     olcUTIL_Geometry2D_py.cpp \
     -o olcUTIL_Geometry2D_py$(python3-config --extension-suffix)
   ```

**Usage**

```python
from olcUTIL_Geometry2D_py import v, circle, rect, line, triangle, contains

p = v(1, 2)
c = circle(v(0, 0), 5)
l = line(v(0, 0), v(1, 1))
r = rect(v(0, 0), v(1, 1))
d = triangle(v(0, 0), v(1, 1), v(0, 1))

if contains(c, p):
   print("Point is inside the circle")
```

**Testing**

The included `test_python.py` file runs the `contains` functionality for all permutations of supported geometric types.

Here is the output of the test_python script:
```
144_geom2d/source01/python $ python test_python.py
circle around p <circle pos=(1.000000,2.000000) radius=0.000000>
circle around c <circle pos=(0.000000,0.000000) radius=5.000000>
circle around l <circle pos=(0.500000,0.500000) radius=0.707107>
circle around r <circle pos=(0.500000,0.500000) radius=0.707107>
circle around d <circle pos=(0.500000,0.500000) radius=0.707107>
bbox around p <rect pos=(1.000000,2.000000) size=(0.000000,0.000000)>
bbox around c <rect pos=(-5.000000,-5.000000) size=(10.000000,10.000000)>
bbox around l <rect pos=(0.000000,0.000000) size=(1.000000,1.000000)>
bbox around r <rect pos=(0.000000,0.000000) size=(1.000000,1.000000)>
bbox around d <rect pos=(0.000000,0.000000) size=(1.000000,1.000000)>
d is not inside r
r is inside d
d is inside l
l is not inside d
r is inside l
l is not inside r
d is not inside c
c is inside d
r is not inside c
c is inside r
l is not inside c
c is inside l
d is not inside p
p is not inside d
r is not inside p
p is not inside r
l is not inside p
p is not inside l
c is inside p
p is not inside c
d intersects r in [<v x=0.000000, y=0.000000>, <v x=1.000000, y=1.000000>, <v x=0.000000, y=1.000000>]
r intersects d in [<v x=0.000000, y=0.000000>, <v x=1.000000, y=1.000000>, <v x=0.000000, y=1.000000>]
d intersects l in [<v x=1.000000, y=1.000000>, <v x=0.000000, y=0.000000>]
l intersects d in [<v x=1.000000, y=1.000000>, <v x=0.000000, y=0.000000>]
r intersects l in [<v x=0.000000, y=0.000000>, <v x=1.000000, y=1.000000>]
l intersects r in [<v x=0.000000, y=0.000000>, <v x=1.000000, y=1.000000>]
d intersects c in []
c intersects d in []
r intersects c in []
c intersects r in []
l intersects c in []
c intersects l in []
d intersects p in []
p intersects d in []
r intersects p in []
p intersects r in []
l intersects p in []
p intersects l in []
c intersects p in []
p intersects c in []
```

**Notes**

* The order of arguments within the `contains` function might not
  affect the results. I shall clarify the intended semantics of this
  function based on the original C++ library.

**Alternative Build Approach 1 (Work in Progress)**

```sh
python setup.py build_ext --inplace 
```
While this method provides a canonical Python build method, it may require further configuration to make it functional. 
On my system a library is compiled but it 

**Alternative Build Approach 2**

```sh
./setup02_cmake.sh
```
This method downloads a specific release of pybind11 and compiles a library using cmake. It works on my system.
