
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
  - components: 
    - abstraction .. minimal data abstraction
	- presentation .. interactive services for end user
	- control .. connected behaviour, non-standard creation, high
      level management
