import olcUTIL_Geometry2D_py as g

p = g.Vector2D(1.0, 2.0)
c = g.Circle(g.Vector2D(0.0, 0.0), 5.0)

if g.contains(c, p):
    print("Point is inside the circle")
