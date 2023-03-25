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
#include <Geom_Curve.hxx>
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
  TopoDS_Shape myBody = BRepPrimAPI_MakePrism(myFaceProfile, aPrismVec);
  auto neckLocation = gp_Pnt(0, 0, myHeight);
  auto neckAxis = gp::DZ();
  auto neckAx2 = gp_Ax2(neckLocation, neckAxis);
  auto myNeckRadius = ((myThickness) / ((4.0)));
  auto myNeckHeight = ((myHeight) / ((10.)));
  auto MKCylinder =
      BRepPrimAPI_MakeCylinder(neckAx2, myNeckRadius, myNeckHeight);
  auto myNeck = MKCylinder.Shape();
  // attach the neck to the body
  myBody = BRepAlgoAPI_Fuse(myBody, myNeck);

  auto mkFillet = ([&]() {
    auto fillet = BRepFilletAPI_MakeFillet(myBody);
    auto edgeExplorer = TopExp_Explorer(myBody, TopAbs_EDGE);
    while (edgeExplorer.More()) {
      auto cur = edgeExplorer.Current();
      auto edge = TopoDS::Edge(cur);
      auto mz = ([&]() {
        auto uStart = Standard_Real(0);
        auto uEnd = Standard_Real(0);
        auto curve = opencascade::handle<Geom_Curve>(
            BRep_Tool::Curve(edge, uStart, uEnd));
        auto N = 100;
        auto deltaU = ((((uEnd) - (uStart))) / ((1.0 * N)));
        auto points = ([&]() {
          auto points = std::vector<gp_Pnt>();
          for (auto i = 0; i < N; i += 1) {
            auto u = (uStart + (deltaU * i));
            points.emplace_back(curve->Value(u));
          }
          return points;
        })();
        auto maxPointIt =
            std::max_element(points.begin(), points.end(),
                             [](auto a, auto b) { return a.Z() < b.Z(); });
        auto maxPoint = *(maxPointIt);
        return maxPoint.Z();
      })();
      std::cout << ""
                << " mz='" << mz << "' " << std::endl;
      // i want to fillet the edge where the neck attaches to the body but not
      // the top of the neck
      if (mz <= 40) {
        fillet.Add(((myThickness) / (12)), edge);
      }
      edgeExplorer.Next();
    }
    return fillet;
  })();
  // make the outside of the body rounder
  myBody = mkFillet.Shape();

  auto facesToRemove = ([&]() {
    auto faceToRemove = TopoDS_Face();
    auto zMax = Standard_Real(-100);
    auto explorer = TopExp_Explorer(myBody, TopAbs_FACE);
    for (; explorer.More(); explorer.Next()) {
      auto aFace = TopoDS::Face(explorer.Current());
      auto bas = BRepAdaptor_Surface(aFace);
      if (GeomAbs_Plane == bas.GetType()) {
        auto plane = bas.Plane();
        auto aPnt = plane.Location();
        if (!(plane.Axis().Direction().IsParallel(
                gp::DZ(), (((1.0 * M_PI)) / ((1.80e+2f)))))) {
          continue;
        }
        auto aZ = aPnt.Z();
        if (zMax < aZ) {
          zMax = aZ;
          faceToRemove = aFace;
        }
      }
    }
    auto facesToRemove = TopTools_ListOfShape();
    facesToRemove.Append(faceToRemove);
    return facesToRemove;
  })();
  // make inside of the bottle hollow
  myBody = ([&]() {
    auto aSolidMaker = BRepOffsetAPI_MakeThickSolid();
    aSolidMaker.MakeThickSolidByJoin(myBody, facesToRemove, -myThickness / 50,
                                     (1.00e-3f));
    return aSolidMaker.Shape();
  })();

  auto aRes = ([&]() {
    auto a = TopoDS_Compound();
    auto b = BRep_Builder();
    b.MakeCompound(a);
    b.Add(a, myBody);
    return a;
  })();
  return aRes;
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
