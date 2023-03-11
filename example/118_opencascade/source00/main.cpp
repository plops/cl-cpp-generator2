#include <GL/glut.h>
#include <memory>
#include <opencascade/Geom_BSplineCurve.hxx>
#include <opencascade/Geom_CartesianPoint.hxx>
#include <opencascade/IGESControl_Reader.hxx>
#include <opencascade/IGESControl_Writer.hxx>
#include <opencascade/ShapeFix_Shape.hxx>
#include <opencascade/ShapeUpgrade_RemoveInternalWires.hxx>
#include <opencascade/ShapeUpgrade_UnifySameDomain.hxx>
#include <opencascade/Standard_Version.hxx>
#include <opencascade/TColgp_HArray1OfPnt.hxx>
#include <opencascade/TopTools_ListOfShape.hxx>
#include <opencascade/TopoDS.hxx>
#include <opencascade/TopoDS_Edge.hxx>
#include <vector>

void display() {
  auto knots = std::vector<double>({0, 1, 2, 3});
  auto knot_multi = std::vector<int>({3, 1, 1, 3});
  auto control_points =
      std::vector<gp_Pnt>({gp_Pnt(0.0, 0.0, 0.0), gp_Pnt(1.0, 2.0, 0.0),
                           gp_Pnt(2.0, -1.0, 0.0), gp_Pnt(3.0, 0.0, 0.0)});
  Handle(TColgp_HArray1OfPnt) points =
      new TColgp_HArray1OfPnt(1, control_points.size());
  for (auto p : control_points) {
    auto i = ((&p) - (&(control_points[0])));
    points->SetValue((i + 1), control_points[i]);
  }
  Handle(Geom_BSplineCurve) curve =
      new Geom_BSplineCurve(points, knots, knot_multi, 3);
}

int main(int argc, char **argv) {
  display();
  return 0;
}
