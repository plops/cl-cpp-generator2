import pygeometry

p = pygeometry.v_2d(1.0, 2.0)
c = pygeometry.circle(pygeometry.v_2d(0.0, 0.0), 5.0)

if pygeometry.contains(c, p):
    print("Point is inside the circle")
