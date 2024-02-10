revise this readme of my github project

- create python binding for https://github.com/OneLoneCoder/olcUTIL_Geometry2D/blob/main/olcUTIL_Geometry2D.h
- the c++ code for the binding is auto generated and it should be easy
  to create instances of all permutations. right now i only expose
  `contains` for all permutations of v_2d, circle, rect, line and
  triangle. supporting ray and the other functions (overlaps,
  intersects, closest, envelope_r, envelope_b, reflects, collision)
  should be straight forward
  
Build the module with:  
```
c++ -O3 -Wall -shared -std=c++20 -fPIC $(python3 -m pybind11 --includes) \
  olcUTIL_Geometry2D_py.cpp \
  -o olcUTIL_Geometry2D_py$(python3-config --extension-suffix)
```


After building the extension, you would be able to use it in Python like this:  
   
```python  
from olcUTIL_Geometry2D_py import v, circle, rect, line, triangle, contains
p=v(1, 2)
c=circle(v(0, 0), 5)
l=line(v(0, 0), v(1, 1))
r=rect(v(0, 0), v(1, 1))
d=triangle(v(0, 0), v(1, 1), v(0, 1))
if ( contains(c, p) ):
    print("Point is inside the circle")
```  

The `test_python.py` file tests contains for all permutations. Here is the output:
```
144_geom2d/source01/python $ ./test_python.py 
d is not inside  r
r is inside d
d is inside l
l is not inside  d
r is inside l
l is not inside  r
d is not inside  c
c is inside d
r is not inside  c
c is inside r
l is not inside  c
c is inside l
d is not inside  p
p is not inside  d
r is not inside  p
p is not inside  r
l is not inside  p
p is not inside  l
c is inside p
p is not inside  c
```
I find it strange that the c is inside p. Perhaps the semantics of `contains` are such that order of the arguments don't matter?


I renamed the type v_2d to v.







An alternative way to build the python module (and the more canonical python way) is to call the C++ compiler directly:

```sh  
python setup.py build_ext --inplace  
```  
   
However, while this compiles on my system, I haven't got it to work
yet.
