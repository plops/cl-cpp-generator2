#include <BRepAlgoAPI_Common.hxx>
#include <BRepAlgoAPI_Cut.hxx>
#include <BRepAlgoAPI_Fuse.hxx>
#include <BRepBuilderAPI_MakeEdge.hxx>
#include <BRepBuilderAPI_MakeFace.hxx>
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

TopoDS_Shape MakeCrownedPulleyFlatShaft(const Standard_Real shaftDiameter,
                                        const Standard_Real centralDiameter,
                                        const Standard_Real pulleyThickness,
                                        const Standard_Real shaftLength,
                                        const Standard_Real flatLength,
                                        const Standard_Real flatThickness) {
  auto sphere = BRepPrimAPI_MakeSphere(gp_Pnt(0, 0, pulleyThickness / 2),
                                       centralDiameter / 2);
  auto axis = gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1));
  auto cylBig =
      BRepPrimAPI_MakeCylinder(axis, centralDiameter / 2, pulleyThickness);
  auto cylShaft = BRepBuilderAPI_Transform(
      BRepPrimAPI_MakeCylinder(axis, shaftDiameter / 2,
                               ((shaftLength) - (flatLength))),
      ([&]() {
        auto a = gp_Trsf();
        a.SetTranslation(gp_Vec(0, 0, flatLength));
        return a;
      })());
  auto cylShaftFullLength =
      BRepPrimAPI_MakeCylinder(axis, shaftDiameter / 2, pulleyThickness);
  auto shaftFlattening = BRepBuilderAPI_Transform(
      BRepPrimAPI_MakeBox(flatThickness, centralDiameter, flatLength), ([&]() {
        auto a = gp_Trsf();
        a.SetTranslation(gp_Vec(-flatThickness / 2, -centralDiameter / 2, 0));
        return a;
      })());
  auto cylShaft2 = BRepAlgoAPI_Common(cylShaftFullLength, shaftFlattening);
  auto disk = BRepAlgoAPI_Common(sphere, cylBig);
  TopoDS_Shape shape =
      BRepAlgoAPI_Cut(disk, BRepAlgoAPI_Fuse(cylShaft2, cylShaft));
  auto unify = ShapeUpgrade_UnifySameDomain(shape);
  // remove unneccessary seams
  unify.Build();
  shape = unify.Shape();

  return shape;
}

TopoDS_Shape MakeCrownedPulley(const Standard_Real shaftDiameter,
                               const Standard_Real centralDiameter,
                               const Standard_Real pulleyThickness) {
  auto sphere = BRepPrimAPI_MakeSphere(gp_Pnt(0, 0, pulleyThickness / 2),
                                       centralDiameter / 2);
  auto axis = gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1));
  auto cylBig =
      BRepPrimAPI_MakeCylinder(axis, centralDiameter / 2, pulleyThickness);
  auto cylShaftFullLength =
      BRepPrimAPI_MakeCylinder(axis, shaftDiameter / 2, pulleyThickness);
  auto disk = BRepAlgoAPI_Common(sphere, cylBig);
  TopoDS_Shape shape = BRepBuilderAPI_Transform(
      BRepAlgoAPI_Cut(disk, cylShaftFullLength), ([&]() {
        auto a = gp_Trsf();
        a.SetTranslation(gp_Vec(25, 0, 0));
        return a;
      })());
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
  auto shape = MakeCrownedPulleyFlatShaft((0.01 + 4.92), (20.f), (8.310f),
                                          (8.310f), (5.870f), (0.01 + 2.94));
  auto label = ST->AddShape(shape, false);
  auto shape2 = MakeCrownedPulley((0.01 + 12.82), (20.f), (8.310f));
  auto label2 = ST->AddShape(shape2, false);

  auto status = app->SaveAs(doc, "doc.xbf");
  if (!(PCDM_SS_OK == status)) {
    return 1;
  }
  WriteStep(doc, "o.stp");

  return 0;
}
