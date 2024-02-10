#!/usr/bin/env python3
from olcUTIL_Geometry2D_py import v, circle, rect, line, triangle, contains, closest, overlaps, intersects, envelope_c, envelope_r
p=v(1, 2)
c=circle(v(0, 0), 5)
l=line(v(0, 0), v(1, 1))
r=rect(v(0, 0), v(1, 1))
d=triangle(v(0, 0), v(1, 1), v(0, 1))
print("circle around p {}".format(envelope_c(p)))
print("circle around c {}".format(envelope_c(c)))
print("circle around l {}".format(envelope_c(l)))
print("circle around r {}".format(envelope_c(r)))
print("circle around d {}".format(envelope_c(d)))
print("bbox around p {}".format(envelope_r(p)))
print("bbox around c {}".format(envelope_r(c)))
print("bbox around l {}".format(envelope_r(l)))
print("bbox around r {}".format(envelope_r(r)))
print("bbox around d {}".format(envelope_r(d)))
if ( contains(d, r) ):
    print("d is inside r")
else:
    print("d is not inside r")
if ( contains(r, d) ):
    print("r is inside d")
else:
    print("r is not inside d")
if ( contains(d, l) ):
    print("d is inside l")
else:
    print("d is not inside l")
if ( contains(l, d) ):
    print("l is inside d")
else:
    print("l is not inside d")
if ( contains(r, l) ):
    print("r is inside l")
else:
    print("r is not inside l")
if ( contains(l, r) ):
    print("l is inside r")
else:
    print("l is not inside r")
if ( contains(d, c) ):
    print("d is inside c")
else:
    print("d is not inside c")
if ( contains(c, d) ):
    print("c is inside d")
else:
    print("c is not inside d")
if ( contains(r, c) ):
    print("r is inside c")
else:
    print("r is not inside c")
if ( contains(c, r) ):
    print("c is inside r")
else:
    print("c is not inside r")
if ( contains(l, c) ):
    print("l is inside c")
else:
    print("l is not inside c")
if ( contains(c, l) ):
    print("c is inside l")
else:
    print("c is not inside l")
if ( contains(d, p) ):
    print("d is inside p")
else:
    print("d is not inside p")
if ( contains(p, d) ):
    print("p is inside d")
else:
    print("p is not inside d")
if ( contains(r, p) ):
    print("r is inside p")
else:
    print("r is not inside p")
if ( contains(p, r) ):
    print("p is inside r")
else:
    print("p is not inside r")
if ( contains(l, p) ):
    print("l is inside p")
else:
    print("l is not inside p")
if ( contains(p, l) ):
    print("p is inside l")
else:
    print("p is not inside l")
if ( contains(c, p) ):
    print("c is inside p")
else:
    print("c is not inside p")
if ( contains(p, c) ):
    print("p is inside c")
else:
    print("p is not inside c")
print("d intersects r in {}".format(intersects(d, r)))
print("r intersects d in {}".format(intersects(r, d)))
print("d intersects l in {}".format(intersects(d, l)))
print("l intersects d in {}".format(intersects(l, d)))
print("r intersects l in {}".format(intersects(r, l)))
print("l intersects r in {}".format(intersects(l, r)))
print("d intersects c in {}".format(intersects(d, c)))
print("c intersects d in {}".format(intersects(c, d)))
print("r intersects c in {}".format(intersects(r, c)))
print("c intersects r in {}".format(intersects(c, r)))
print("l intersects c in {}".format(intersects(l, c)))
print("c intersects l in {}".format(intersects(c, l)))
print("d intersects p in {}".format(intersects(d, p)))
print("p intersects d in {}".format(intersects(p, d)))
print("r intersects p in {}".format(intersects(r, p)))
print("p intersects r in {}".format(intersects(p, r)))
print("l intersects p in {}".format(intersects(l, p)))
print("p intersects l in {}".format(intersects(p, l)))
print("c intersects p in {}".format(intersects(c, p)))
print("p intersects c in {}".format(intersects(p, c)))