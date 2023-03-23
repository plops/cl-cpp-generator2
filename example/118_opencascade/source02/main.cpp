#include <BRepAlgoAPI_Fuse.hxx>
#include <BRepBuilderAPI_MakeEdge.hxx>
#include <BRepBuilderAPI_MakeFace.hxx>
#include <BRepBuilderAPI_MakeWire.hxx>
#include <BRepBuilderAPI_Transform.hxx>
#include <BRepFilletAPI_MakeFillet.hxx>
#include <BRepLib.hxx>
#include <BRepOffsetAPI_MakeThickSolid.hxx>
#include <BRepOffsetAPI_ThruSections.hxx>
#include <BRepPrimAPI_MakeCylinder.hxx>
#include <BRepPrimAPI_MakePrism.hxx>
#include <BRep_Tool.hxx>
#include <BinXCAFDrivers.hxx>
#include <GCE2d_MakeSegment.hxx>
#include <GC_MakeArcOfCircle.hxx>
#include <GC_MakeSegment.hxx>
#include <Geom2d_Ellipse.hxx>
#include <Geom2d_TrimmedCurve.hxx>
#include <Geom_CylindricalSurface.hxx>
#include <Geom_Plane.hxx>
#include <Geom_Surface.hxx>
#include <Geom_TrimmedCurve.hxx>
#include <STEPCAFControl_Writer.hxx>
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

TopoDS_Shape MakeBottle(const Standard_Real myWidth,
                        const Standard_Real myHeight,
                        const Standard_Real myThickness) {
  auto p1 = gp_Pnt(-myWidth / 2, 0, 0);
  auto p2 = gp_Pnt(-myWidth / 2, -myThickness / 4, 0);
  auto p3 = gp_Pnt(0, -myThickness / 2, 0);
  auto p4 = gp_Pnt(myWidth / 2, -myThickness / 4, 0);
  auto p5 = gp_Pnt(myWidth / 2, 0, 0);
  auto anArcOfCircle =
      opencascade::handle<Geom_TrimmedCurve>(GC_MakeArcOfCircle(p2, p3, p4));
  auto aSegment1 =
      opencascade::handle<Geom_TrimmedCurve>(GC_MakeSegment(p1, p2));
  auto aSegment2 =
      opencascade::handle<Geom_TrimmedCurve>(GC_MakeSegment(p4, p5));
  auto anEdge1 = BRepBuilderAPI_MakeEdge(aSegment1);
  auto anEdge2 = BRepBuilderAPI_MakeEdge(anArcOfCircle);
  auto anEdge3 = BRepBuilderAPI_MakeEdge(aSegment2);
  auto aWire = BRepBuilderAPI_MakeWire(anEdge1, anEdge2, anEdge3);
  auto aTrsf = ([]() {
    auto xAxis = gp::OX();
    auto a = gp_Trsf();
    a.SetMirror(xAxis);
    return a;
  })();
  auto aBRepTrsf = BRepBuilderAPI_Transform(aWire, aTrsf);
  auto aMirroredShape = aBRepTrsf.Shape();
  auto aMirroredWire = TopoDS::Wire(aMirroredShape);
  auto mkWire = ([&]() {
    auto a = BRepBuilderAPI_MakeWire();
    a.Add(aWire);
    a.Add(aMirroredWire);
    return a;
  })();
  auto myWireProfile = mkWire.Wire();
  auto myFaceProfile = BRepBuilderAPI_MakeFace(myWireProfile);
  auto aPrismVec = gp_Vec(0, 0, myHeight);
  auto myBody = BRepPrimAPI_MakePrism(myFaceProfile, aPrismVec);
  auto mkFillet = ([&]() {
    auto fillet = BRepFilletAPI_MakeFillet(myBody);
    auto edgeExplorer = TopExp_Explorer(myBody, TopAbs_EDGE);
    while (edgeExplorer.More()) {
      auto edge = TopoDS::Edge(edgeExplorer.Current());
      fillet.Add(((myThickness) / (12)), edge);
      edgeExplorer.Next();
    }
    return fillet;
  })();
  auto myBody1 = mkFillet.Shape();
  auto facesToRemove = ([&]() {
    auto faceToRemove = TopoDS_Face();
    auto zMax = Standard_Real(-100);
    auto explorer = TopExp_Explorer(myBody1, TopAbs_FACE);
    for (; explorer.More(); explorer.Next()) {
      auto aFace = TopoDS::Face(explorer.Current());
      auto bas = BRepAdaptor_Surface(aFace);
      std::cout << "face"
                << " bas.GetType()='" << bas.GetType() << "' "
                << " GeomAbs_Plane='" << GeomAbs_Plane << "' " << std::endl;
      if (GeomAbs_Plane == bas.GetType()) {
        auto plane = bas.Plane();
        auto aPnt = plane.Location();
        if (!(plane.Axis().Direction().IsParallel(
                gp::DZ(), (((1.0 * M_PI)) / ((1.80e+2f)))))) {
          continue;
        }
        auto aZ = aPnt.Z();
        if (zMax < aZ) {
          std::cout << ""
                    << " zMax='" << zMax << "' "
                    << " aZ='" << aZ << "' " << std::endl;
          zMax = aZ;
          faceToRemove = aFace;
        }
      }
    }
    auto facesToRemove = TopTools_ListOfShape();
    facesToRemove.Append(faceToRemove);
    return facesToRemove;
  })();
  auto myBody2 = ([&]() {
    auto aSolidMaker = BRepOffsetAPI_MakeThickSolid();
    aSolidMaker.MakeThickSolidByJoin(myBody1, facesToRemove, -myThickness / 50,
                                     (1.00e-3f));
    return aSolidMaker.Shape();
  })();
  auto aRes = ([&]() {
    auto a = TopoDS_Compound();
    auto b = BRep_Builder();
    b.MakeCompound(a);
    b.Add(a, myBody2);
    return a;
  })();
  return aRes;
}

bool WriteStep(const Handle(TDocStd_Document) & doc, const char *filename) {
  auto Writer = STEPCAFControl_Writer();
  if (!(Writer.Transfer(doc))) {
    return false;
  }
  if (!(IFSelect_RetDone == Writer.Write(filename))) {
    return false;
  }
  return true;
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
  auto W = (30.);
  auto H = (40.);
  auto T = (10.);
  auto shape = MakeBottle(W, H, T);
  auto label = ST->AddShape(shape, false);

  auto status = app->SaveAs(doc, "doc.xbf");
  if (!(PCDM_SS_OK == status)) {
    return 1;
  }
  WriteStep(doc, "o.stp");

  return 0;
}