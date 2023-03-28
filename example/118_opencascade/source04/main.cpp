#include <BRepAlgoAPI_Common.hxx>
#include <BRepAlgoAPI_Cut.hxx>
#include <BRepAlgoAPI_Fuse.hxx>
#include <BRepBuilderAPI_MakeEdge.hxx>
#include <BRepBuilderAPI_MakeFace.hxx>
#include <BRepBuilderAPI_MakePolygon.hxx>
#include <BRepBuilderAPI_MakeWire.hxx>
#include <BRepBuilderAPI_Transform.hxx>
#include <BRepFilletAPI_MakeFillet.hxx>
#include <BRepLib.hxx>
#include <BRepOffsetAPI_MakeThickSolid.hxx>
#include <BRepOffsetAPI_ThruSections.hxx>
#include <BRepPrimAPI_MakeBox.hxx>
#include <BRepPrimAPI_MakeCylinder.hxx>
#include <BRepPrimAPI_MakePrism.hxx>
#include <BRepPrimAPI_MakeRevol.hxx>
#include <BRepPrimAPI_MakeSphere.hxx>
#include <BRep_Tool.hxx>
#include <BinXCAFDrivers.hxx>
#include <GCE2d_MakeSegment.hxx>
#include <GC_MakeArcOfCircle.hxx>
#include <GC_MakeSegment.hxx>
#include <Geom2d_Ellipse.hxx>
#include <Geom2d_Parabola.hxx>
#include <Geom2d_TrimmedCurve.hxx>
#include <Geom_Curve.hxx>
#include <Geom_CylindricalSurface.hxx>
#include <Geom_Parabola.hxx>
#include <Geom_Plane.hxx>
#include <Geom_Surface.hxx>
#include <Geom_TrimmedCurve.hxx>
#include <STEPCAFControl_Writer.hxx>
#include <ShapeUpgrade_UnifySameDomain.hxx>
#include <TDocStd_Application.hxx>
#include <TopExp_Explorer.hxx>
#include <TopTools_ListOfShape.hxx>
#include <TopoDS_Compound.hxx>
#include <TopoDS_Edge.hxx>
#include <TopoDS_Face.hxx>
#include <TopoDS_Shape.hxx>
#include <TopoDS_Wire.hxx>
#include <XCAFDoc_DocumentTool.hxx>
#include <XCAFDoc_ShapeTool.hxx>
#include <algorithm>
#include <gp.hxx>
#include <gp_Ax1.hxx>
#include <gp_Ax2.hxx>
#include <gp_Ax2d.hxx>
#include <gp_Dir.hxx>
#include <gp_Dir2d.hxx>
#include <gp_Pnt.hxx>
#include <gp_Pnt2d.hxx>
#include <gp_Trsf.hxx>
#include <gp_Vec.hxx>
#include <iostream>
#include <vector>
// https://en.wikipedia.org/wiki/ISO_metric_screw_thread
// https://dev.opencascade.org/doc/overview/html/occt__tutorial.html

TopoDS_Shape MakeHolder() {
  auto axis = gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1));
  auto thick = (((5.0f)) - ((1.00e-2f)));
  auto adapterRad =
      ((0.50f) * ((29.490f) + (5.00e-2f)) * (((30.020f)) / ((29.30f))));
  auto centralCylOut = BRepPrimAPI_MakeCylinder(axis, (adapterRad + 5), thick);
  auto centralCylIn = BRepPrimAPI_MakeCylinder(axis, adapterRad, thick);
  auto centralCylClearance = BRepBuilderAPI_Transform(
      BRepPrimAPI_MakeCylinder(
          axis, ((0.50f) * ((30.040f) + (0.20f)) * (((30.020f)) / ((29.30f)))),
          20),
      ([&]() {
        auto a = gp_Trsf();
        a.SetTranslation(gp_Vec(0, 0, thick));
        return a;
      })());
  auto motorRadBottom = ((0.50f) * ((27.940f) + (4.00e-2f)));
  auto motorRadMid = ((0.50f) * ((28.620f) + (4.00e-2f)));
  auto leftMotorShiftX = -31;
  auto leftMotorHoleBottom = BRepBuilderAPI_Transform(
      BRepPrimAPI_MakeCylinder(axis, motorRadBottom, thick), ([&]() {
        auto a = gp_Trsf();
        a.SetTranslation(gp_Vec(leftMotorShiftX, 0, 0));
        return a;
      })());
  auto leftMotorHoleMid = BRepBuilderAPI_Transform(
      BRepPrimAPI_MakeCylinder(axis, motorRadMid, 24), ([&]() {
        auto a = gp_Trsf();
        a.SetTranslation(gp_Vec(leftMotorShiftX, 0, (1.40f)));
        return a;
      })());
  auto leftMotorBlockMid =
      BRepBuilderAPI_Transform(BRepPrimAPI_MakeBox(20, 40, 12), ([&]() {
                                 auto a = gp_Trsf();
                                 a.SetTranslation(gp_Vec(-55, -20, (1.40f)));
                                 return a;
                               })());
  auto leftMotorWall = BRepBuilderAPI_Transform(
      BRepPrimAPI_MakeCylinder(axis, (3 + motorRadMid), 5), ([&]() {
        auto a = gp_Trsf();
        a.SetTranslation(gp_Vec(leftMotorShiftX, 0, 0));
        return a;
      })());
  auto leftPostHeight = ((4.120f) + (((19.320f)) - ((0.830f)) - ((4.00e-2f))));
  auto leftScrewPostNorth = BRepBuilderAPI_Transform(
      BRepPrimAPI_MakeCylinder(axis, (3.50f), leftPostHeight), ([&]() {
        auto a = gp_Trsf();
        a.SetTranslation(gp_Vec(leftMotorShiftX, ((35) / (2)), 0));
        return a;
      })());
  auto leftScrewPostHoleNorth = BRepBuilderAPI_Transform(
      BRepPrimAPI_MakeCylinder(axis, ((0.50f) * (2.930f)), leftPostHeight),
      ([&]() {
        auto a = gp_Trsf();
        a.SetTranslation(gp_Vec(leftMotorShiftX, ((35) / (2)), 0));
        return a;
      })());
  auto leftScrewPostSouth = BRepBuilderAPI_Transform(
      BRepPrimAPI_MakeCylinder(axis, (3.50f), leftPostHeight), ([&]() {
        auto a = gp_Trsf();
        a.SetTranslation(gp_Vec(leftMotorShiftX, ((-35) / (2)), 0));
        return a;
      })());
  auto leftScrewPostHoleSouth = BRepBuilderAPI_Transform(
      BRepPrimAPI_MakeCylinder(axis, ((0.50f) * (2.930f)), leftPostHeight),
      ([&]() {
        auto a = gp_Trsf();
        a.SetTranslation(gp_Vec(leftMotorShiftX, ((-35) / (2)), 0));
        return a;
      })());
  auto rightMotorShiftX = (-(leftMotorShiftX));
  auto rightMotorWall = BRepBuilderAPI_Transform(
      BRepPrimAPI_MakeCylinder(axis, (3 + motorRadMid), 5), ([&]() {
        auto a = gp_Trsf();
        a.SetTranslation(gp_Vec(rightMotorShiftX, 0, 0));
        return a;
      })());
  auto rightPostHeight = ((((9.420f) + (7.90f) + 8)) - ((6.760f)));
  auto rightMotorHoleMid = BRepBuilderAPI_Transform(
      BRepPrimAPI_MakeCylinder(axis, motorRadMid, rightPostHeight), ([&]() {
        auto a = gp_Trsf();
        a.SetTranslation(gp_Vec(rightMotorShiftX, 0, 0));
        return a;
      })());
  auto rightScrewPostNorth = BRepBuilderAPI_Transform(
      BRepPrimAPI_MakeCylinder(axis, (3.50f), rightPostHeight), ([&]() {
        auto a = gp_Trsf();
        a.SetTranslation(gp_Vec(rightMotorShiftX, ((35) / (2)), 0));
        return a;
      })());
  auto rightScrewPostSouth = BRepBuilderAPI_Transform(
      BRepPrimAPI_MakeCylinder(axis, (3.50f), rightPostHeight), ([&]() {
        auto a = gp_Trsf();
        a.SetTranslation(gp_Vec(rightMotorShiftX, ((-35) / (2)), 0));
        return a;
      })());
  auto rightScrewPostHoleNorth = BRepBuilderAPI_Transform(
      BRepPrimAPI_MakeCylinder(axis, ((0.50f) * (2.930f)), rightPostHeight),
      ([&]() {
        auto a = gp_Trsf();
        a.SetTranslation(gp_Vec(rightMotorShiftX, ((35) / (2)), 0));
        return a;
      })());
  auto rightScrewPostHoleSouth = BRepBuilderAPI_Transform(
      BRepPrimAPI_MakeCylinder(axis, ((0.50f) * (2.930f)), rightPostHeight),
      ([&]() {
        auto a = gp_Trsf();
        a.SetTranslation(gp_Vec(rightMotorShiftX, ((-35) / (2)), 0));
        return a;
      })());
  auto rightMotorHoleMidFill = BRepBuilderAPI_Transform(
      BRepAlgoAPI_Cut(
          BRepPrimAPI_MakeCylinder(axis, ((motorRadMid) - ((0.10f))),
                                   ((rightPostHeight) - ((1.40f)))),
          BRepPrimAPI_MakeCylinder(axis, ((motorRadMid) - ((0.10f)) - (2)),
                                   ((rightPostHeight) - ((1.40f))))),
      ([&]() {
        auto a = gp_Trsf();
        a.SetTranslation(gp_Vec(0, 40, 0));
        return a;
      })());
  TopoDS_Shape shape = BRepAlgoAPI_Cut(
      BRepAlgoAPI_Fuse(
          centralCylOut,
          BRepAlgoAPI_Fuse(
              rightMotorWall,
              BRepAlgoAPI_Fuse(
                  BRepAlgoAPI_Cut(rightScrewPostSouth, rightScrewPostHoleSouth),
                  BRepAlgoAPI_Fuse(
                      BRepAlgoAPI_Cut(rightScrewPostNorth,
                                      rightScrewPostHoleNorth),
                      BRepAlgoAPI_Fuse(
                          BRepAlgoAPI_Cut(leftScrewPostSouth,
                                          leftScrewPostHoleSouth),
                          BRepAlgoAPI_Fuse(
                              leftMotorWall,
                              BRepAlgoAPI_Cut(leftScrewPostNorth,
                                              leftScrewPostHoleNorth))))))),
      BRepAlgoAPI_Fuse(
          centralCylClearance,
          BRepAlgoAPI_Fuse(
              centralCylIn,
              BRepAlgoAPI_Fuse(
                  rightMotorHoleMid,
                  BRepAlgoAPI_Fuse(leftMotorBlockMid,
                                   BRepAlgoAPI_Fuse(leftMotorHoleBottom,
                                                    leftMotorHoleMid))))));
  auto unify = ShapeUpgrade_UnifySameDomain(shape);
  // remove unneccessary seams
  unify.Build();
  shape = unify.Shape();

  return shape;
}

bool WriteStep(const Handle(TDocStd_Document) & doc, const char *filename) {
  auto Writer = STEPCAFControl_Writer();
  if (!(Writer.Transfer(doc))) {
    return false;
  }
  return IFSelect_RetDone == Writer.Write(filename);
}

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;
  auto app = opencascade::handle<TDocStd_Application>(new TDocStd_Application);
  BinXCAFDrivers::DefineFormat(app);
  auto doc = opencascade::handle<TDocStd_Document>();
  app->NewDocument("BinXCAF", doc);
  auto ST = opencascade::handle<XCAFDoc_ShapeTool>(
      XCAFDoc_DocumentTool::ShapeTool(doc->Main()));
  auto shape = MakeHolder();
  auto label = ST->AddShape(shape, false);

  auto status = app->SaveAs(doc, "doc.xbf");
  if (!(PCDM_SS_OK == status)) {
    return 1;
  }
  WriteStep(doc, "o.stp");

  return 0;
}
