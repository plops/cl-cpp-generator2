**olcUTIL_Geometry2D: Python Bindings for 2D Geometry Calculations**

**About**

This project provides easy-to-use Python bindings for the 2D geometry
functions defined in the [olcUTIL_Geometry2D.h]
([https://github.com/OneLoneCoder/olcUTIL_Geometry2D/blob/main/olcUTIL_Geometry2D.h](https://github.com/OneLoneCoder/olcUTIL_Geometry2D/blob/main/olcUTIL_Geometry2D.h))
C++ header. Simplify your geometric calculations within Python using
these bindings.

**Features**

* **Core Functionality:** Currently exposes the `contains` function for all combinations of the following geometric types: 
    *  `v` (formerly `v_2d`): Represents a 2D vector/point.
    * `circle` 
    * `rect` (rectangle)
    * `line`
    * `triangle`
* **Planned Expansion:** Support for the following functions from olcUTIL_Geometry2D is intended:
    * `overlaps`
    * `intersects`
    * `closest`
    * `envelope_r`
    * `envelope_b`
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

The included `test_python.py` file validates the `contains` functionality for all permutations of supported geometric types.

**Notes**

* The order of arguments within the `contains` function might not
  affect the results. I shall clarify the intended semantics of this
  function based on the original C++ library.

**Alternative Build Approach (Work in Progress)**

```sh
python setup.py build_ext --inplace 
```
While this method provides a canonical Python build method, it may require further configuration to make it functional. 
