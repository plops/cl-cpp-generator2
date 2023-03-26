
|       |                                             |
| gen00 | define a spline                             |
| gen01 | create a car chassis, color the wheel faces |
| gen02 | model a bottle                              |
| gen03 | model a pulley to be used with 120  pico pi |
|       |                                             |

https://ftp.fau.de/fosdem/2020/H.2213/opencascade.webm

http://analysissitus.org/forum/index.php?threads/youtube-lessons.3/

bottle tutorial
https://dev.opencascade.org/doc/overview/html/occt__tutorial.html
https://youtu.be/kPcD5liq8Cs
https://github.com/3drepo/occt/blob/master/samples/qt/Tutorial/src/MakeBottle.cxx

cmake
https://www.youtube.com/watch?v=SfRFG_Pk9pk


I found this description of 'almost always
auto'. https://www.quora.com/Is-it-good-practice-to-declare-most-variables-using-auto-in-C
This is the style I use in my generated C++ code.


disconnect part from plate in freezer
https://www.youtube.com/watch?v=Jo1aoHnRY5Q

how to design spur gears
https://www.youtube.com/watch?v=8bml2pK6Ra0

https://www.youtube.com/watch?v=IBcGLpQnfYk

https://www.youtube.com/watch?v=gt_Ofn95ML0 involute f


- freecad to convert step to stl
 https://youtu.be/v6FgTIpsCKo
 
 mesh design
 meshes -> create mesh from shape
 

- website with opencascade
http://www.creativecadtechnology.com/OCC/ShowScript?userName=learnMoreAboutOCC&groupName=demo&scriptName=MakeBottle


# presentation with 180 slides about opencascade
- http://slideplayer.com/slide/5112213/
- i think this presentation is what was missing. it gives an overview
  about the class naming conventions and the design methology that
  underlies opencascade
	
- handles
  - DynamicType, IsInstance, IsKind
  - IsNull, Nullify
  - DownCast
- geometry
  - analytic vs parametric
  - conic, bezier, b-spline
  - interpolation vs approximation
  - approximation: tangency, curvature, tolerance constraints
  - geometry: representation of simple shapes which have a
    mathematical description
  - topology: delimit geometric region by boundaries, group regions to
    define complex shapes
	
- elementary geometry
  - components are in separate packages: 
    - abstraction .. minimal data abstraction
	  - perennial .. new controls can be added without changing the
        abstraction
      - minimal .. controls are created as needed during the session
	  - controls have a lifetime, intermediate results can be queried
        before validation
	- presentation .. interactive services for end user
	- control .. connected behaviour, non-standard creation, high
      level management
  - example 2d circle:
    - abstraction: axis and radius
	- control: build circle from center and radius, build circle
      through 3 points
    - presentation: the object you display in a viewer
  - applied controls in open cascade can be:
	- direct construction
	- constraint constructions in 2D
	- algorithms (intersect, boolean operations)
	
- basic geometry packages
  - manipulated by value, no inheritance
  - data abstraction entity defintion `gp`: gp_Pnt, gp_Vec, gp_Lin2d,
      gp_Circ
  - controls direct construction `gce`: gce_MakeCircle, gce_MakeLin2d
  - constrained construction in 2D `GccAna`: GccAna_Circ2d2TanRad

  - 2D gp: 
	- XY, Pnt2d, Mat2d, Trsf2d, GTrsf2d
	- Vec2d, Dir2d, Ax2d, Lin2d, Circ2d, Elips2d, Hypr2d, Parab2d
	- Ax22d
	
  - 3D gp:
	- XYZ, Pnt, Mat, Trsf, GTrsf
	- Vec, Dir, Ax1, Lin, Circ, Elips, Hypr, Parab
	- Ax2, Ax3, Pin
	- Cylinder, Sphere, Torus
	
- advanced geometry
 - 3D
   - abstraction entities definition: Geom_BezierCurve
   - controls direct construction: GC_MakeTrimmedCylinder
 - 2d
   - entities definition: Geom2d_BoundedCurve
   - controls direct construction: GCE2d_MakeArcOfParabola
   - constrained construction: Geom2DGcc_Circ2d3Tan
   
 - hierarchy of classes is STEP compliant
 - tools allow to go back and forth from Geom to gp:

```
Handle(Geom_Circle) C = new Geom_Circle(gp_Circ c);
gp_Circ c = C->Circ();
```
 - entities from GC, GCE2d and Geom2dGcc are manipulated by value
   (control classes)

- Geom_Geometry
 - Geom_Point
   - Geom_CartesianPoint
 - Geom_Vector
   - Geom_Direction
   - Geom_VectorWithMagnitude
 - Geom_Curve
   - Geom_BoundedCurve
     - Geom_BSplineCurve
	 - Geom_BezierCurve
	 - Geom_TrimmedCurve
   - Geom_Conic
     - Geom_Circle
	 - Geom_Ellipse
	 - Geom_Hyperbola
	 - Geom_Parabola
   - Geom_Line
   - Geom_OffsetCurve
 - Geom_Surface
   - Geom_BoundedSurface
     - Geom_BSplineSurface
	 - Geom_BezierSurface
	 - Geom_RectangularTrimmedSurface
   - Geom_SweptSurface
     - Geom_SurfaceOfLinearExtrusion
	 - Geom_SurfaceOfRevolution
   - Geom_OffsetSurface
   - Geom_PlateSurface
   - Geom_ElementarySurface
     - Geom_ConicalSurface
	 - Geom_CylindricalSurface
	 - Geom_Plane
	 - Geom_SphericalSurface
	 - Geom_ToroidalSurface
	 

- constraint geometry
  - solution and argument are `outside` each other
  - solution encompasses the argument `enclosing`
  - solution is encompassed by the argument `enclosed`
  - arguments are ordered, this gives implicit orientation
	- by convention, the interior of a countour is on the left
      according to the positive direction of the contour description
	  
	  
- geometry digest

| basic         | adv           | adv         | explanation                                         |
| 2d & 3d       | adv 2d        | adv 3d      |                                                     |
|---------------|---------------|-------------|-----------------------------------------------------|
| gp            | Geom2d*       | Geom*       | basic entities                                      |
| TColgp        | TColGeom2d*   | TColGeom*   | collection of basic entities                        |
| gce           | GCE2d         | GC          | direct construction (1 solution)                    |
| GccAna        | Geom2dGcc     |             | constrained construction (n solutions)              |
|               | Geom2dAPI     | GeomAPI     | projections, extrema, intersections, approximations |
| Adaptor2d, 3d | Geom2dAdaptor | GeomAdaptor | entities and computation info                       |
| CPnts         | GCPnts        | GCPnts      | points on a curve                                   |
| LProp         | Geom2dLProp   | GeomLProp   | local properties                                    |
| Convert       | Geom2dConvert | GeomConvert | Formalism, conversion                               |
|               |               |             |                                                     |

- with star (*) .. manipulated by handle

	- GeomAbs: enumeration for geometric algorithms (curve-type, surface-type, continuity)
	- GccEnt: conversion to qualified line/circle for GccAna
	- Precision: standard precision values (points angles)
	- GeomTools: dump, read, write

- points
  - gp_Pnt(X,Y,Z)
  - gp::Origin()
  - from points: 
    - gp_Pnt::Barycenter of 2 points
	- Translate
	- Translated
	- Rotate
	- Rotated
	- Gprop_PEquation::Point .. mean of a collection of points,
      considered to be coincident
	  
- curve
  - gp_Circle::Location .. center of a circle
  - GCPnts, CPnts .. compute points on a 2D or 3D curve
  - LProp_CLProps .. compute local point on a curve
  - Geom_Curve::D0.. compute point by parameter
  - GeomAPI_ProjectPointOnCurve
  - GeomAPI_ProjectPointOnSurf
  - GeomAPI_IntCS .. from intersections

  - build curve from points
	- FairCurve_Batten: simulate physical splines with constant or
      linearly increasing section
    - GeomAPI_PointsToBSpline: 3d bspline curve that approximates a
      set of points

  - project curve
    - onto plane GeomAPI::To{2,3}d
	- onto any surface: GeomProjLib::Project
	
  - intersection of curves and surfaces
	- Geom2dAPI_InterCurveCurve 
	- GeomAPI_IntSS

  - extrema
	- GeomAPI_ExtremaCurveCurve
	- GeomAPI_ExtremaCurveSurface
	
  - extrapolation extend a bounded curve to a point
	- GeomLib::ExtentCurveToPoint

  - Information
	- local properties of curve: GeomLProp_CurveTool
	

- surfaces (p. 56)
  - from points
  - from curves
  - extremas
  - extrapolation extend bounded surface along one of its boundaries
  - information

- topology
  - topological entities are called shapes in opencascade
  - vertex, wire, face, shell (faces connected by their edges), solid
    (part of space limited by shells), compound (group of any type of
    topological objects), compsolid (solids connected by their faces)
	
  - create shape
	- abstract topology (TopoDS) references to an object, e.g. edge
      described by two vertices
	- boundary representation (edge on a curve and bounded by two
      vertices)
	  
	  
| abstraction                            | algo                                | tools                          |
|----------------------------------------|-------------------------------------|--------------------------------|
| TopoDS abstract topology datastructure | direct construction                 | BRepTools                      |
|                                        | BRep{Builder,Prim,Offset,Fillet}API |                                |
|                                        |                                     | TopExp explore graph of shapes |
| BRep geometric boundary representation | BrepAlogAPI (boolean)               | BRepFeat (modeling features)   |
|                                        |                                     |                                |
