#!/usr/bin/env python3
from olcUTIL_Geometry2D_py import v, circle, rect, line, triangle, contains
p=v(1, 2)
c=circle(v(0, 0), 5)
l=line(v(0, 0), v(1, 1))
r=rect(v(0, 0), v(1, 1))
d=triangle(v(0, 0), v(1, 1), v(0, 1))
if ( contains(d, r) ):
    print("d is inside r")
else:
    print("d is not inside  r")
if ( contains(r, d) ):
    print("r is inside d")
else:
    print("r is not inside  d")
if ( contains(d, l) ):
    print("d is inside l")
else:
    print("d is not inside  l")
if ( contains(l, d) ):
    print("l is inside d")
else:
    print("l is not inside  d")
if ( contains(r, l) ):
    print("r is inside l")
else:
    print("r is not inside  l")
if ( contains(l, r) ):
    print("l is inside r")
else:
    print("l is not inside  r")
if ( contains(d, c) ):
    print("d is inside c")
else:
    print("d is not inside  c")
if ( contains(c, d) ):
    print("c is inside d")
else:
    print("c is not inside  d")
if ( contains(r, c) ):
    print("r is inside c")
else:
    print("r is not inside  c")
if ( contains(c, r) ):
    print("c is inside r")
else:
    print("c is not inside  r")
if ( contains(l, c) ):
    print("l is inside c")
else:
    print("l is not inside  c")
if ( contains(c, l) ):
    print("c is inside l")
else:
    print("c is not inside  l")
if ( contains(d, p) ):
    print("d is inside p")
else:
    print("d is not inside  p")
if ( contains(p, d) ):
    print("p is inside d")
else:
    print("p is not inside  d")
if ( contains(r, p) ):
    print("r is inside p")
else:
    print("r is not inside  p")
if ( contains(p, r) ):
    print("p is inside r")
else:
    print("p is not inside  r")
if ( contains(l, p) ):
    print("l is inside p")
else:
    print("l is not inside  p")
if ( contains(p, l) ):
    print("p is inside l")
else:
    print("p is not inside  l")
if ( contains(c, p) ):
    print("c is inside p")
else:
    print("c is not inside  p")
if ( contains(p, c) ):
    print("p is inside c")
else:
    print("p is not inside  c")