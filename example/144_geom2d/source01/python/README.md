- create python binding for https://github.com/OneLoneCoder/olcUTIL_Geometry2D/blob/main/olcUTIL_Geometry2D.h

  
Build the module with:  
   
```sh  
python setup.py build_ext --inplace  
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

I renamed the type v_2d to v.




An alternative way to build the python module is to call the C++ compiler directly:

```
c++ -O3 -Wall -shared -std=c++20 -fPIC $(python3 -m pybind11 --includes) \
  olcUTIL_Geometry2D_py.cpp \
  -o olcUTIL_Geometry2D_py$(python3-config --extension-suffix)
```
