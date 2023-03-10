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
  auto knots = std::vector<double>({0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0});
  auto control_points =
      std::vector<gp_Pnt>({gp_Pnt(0.0, 0.0, 0.0), gp_Pnt(1.0, 2.0, 0.0),
                           gp_Pnt(2.0, -1.0, 0.0), gp_Pnt(3.0, 0.0, 0.0)});
  auto points = std::make_shared<TColgp_HArray1OfPnt>(1, control_points.size());
  for (auto i = 0; i < control_points.size(); i += 1) {
    points->SetValue((1 + i), control_points[i]);
  }
}

int main(int argc, char **argv) {
  display();
  return 0;
}
