#include <BRepAlgoAPI_Fuse.hxx>
#include <BRepBuilderAPI_MakeEdge.hxx>
#include <BRepBuilderAPI_MakeFace.hxx>
#include <BRepBuilderAPI_MakeWire.hxx>
#include <BRepBuilderAPI_Transform.hxx>
#include <BRep_Tool.hxx>
#include <include<>>

TopoDS_Shape BuildAxle(const double D, const double L) {
  return BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(-L / 2, 0, 0), gp::DX()),
                                  ((D) / (2)), L);
}

TopoDS_Shape BuildWheelAxle(const TopoDS_Shape &wheel, const TopoDS_Shape &axle,
                            const double L) {
  auto comp = TopoDS_Compound();
  auto bbuilder = BRep_Builder();
  auto wheelT_right = gp_Trsf();
  wheelT_right.SetTranslationPart(gp_Vec(L / 2, 0, 0));
  auto qn = gp_Quaternion(gp::DY(), M_PI);
  auto Ry = gp_Trsf();
  Ry.SetRotation(qn);
  auto wheelT_left = (wheelT_right.Inverted() * Ry);

  bbuilder.MakeCompound(comp);
  bbuilder.Add(comp, wheel.Moved(wheelT_right));
  bbuilder.Add(comp, wheel.Moved(wheelT_left));
  bbuilder.Add(comp, axle);
  return comp;
}

TopoDS_Shape BuildChassis(const TopoDS_Shape &wheelAxle, const double CL) {
  auto comp = TopoDS_Compound();
  auto bbuilder = BRep_Builder();
  auto frontT = gp_Trsf();
  auto rearT = gp_Trsf();
  frontT.SetTranslationPart(gp_Vec(0, CL / 2, 0));
  rearT.SetTranslationPart(gp_Vec(0, -CL / 2, 0));
  bbuilder.MakeCompound(comp);
  bbuilder.Add(comp, wheelAxle.Moved(frontT));
  bbuilder.Add(comp, wheelAxle.Moved(rearT));
  return comp;
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
class t_prototype {
public:
  TopoDS_Shape shape;
  TDF_Label label;
};
class t_wheelPrototype : public t_prototype {
public:
  TopoDS_Face frontFace;
  TDF_Label frontFaceLabel;
};

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;
  auto app = opencascade::handle<TDocStd_Application>(new TDocStd_Application);
  BinXCAFDrivers::DefineFormat(app);
  auto doc = opencascade::handle<TDocStd_Document>();
  app->NewDocument("BinXCAF", doc);
  auto ST = opencascade::handle<XCAFDoc_ShapeTool>(
      XCAFDoc_DocumentTool::ShapeTool(doc->Main()));
  auto CT = opencascade::handle<XCAFDoc_ColorTool>(
      XCAFDoc_DocumentTool::ColorTool(doc->Main()));
  auto OD = (5.00e+2);
  auto W = (1.00e+2);
  auto D = (50.);
  auto L = (5.00e+2);
  auto CL = (6.00e+2);
  auto wheelProto = t_wheelPrototype();
  wheelProto.shape = BuildWheel(OD, W);
  wheelProto.label = ST->AddShape(wheelProto.shape, false);

  TDataStd_Name::Set(wheelProto.label, "wheel");
  CT->SetColor(wheelProto.label, Quantity_Color(1, 0, 0, Quantity_TOC_RGB),
               XCAFDoc_ColorGen);
  auto axleProto = t_prototype();
  axleProto.shape = BuildAxle(D, L);
  axleProto.label = ST->AddShape(axleProto.shape, false);

  TDataStd_Name::Set(axleProto.label, "axle");
  CT->SetColor(axleProto.label, Quantity_Color(0, 1, 0, Quantity_TOC_RGB),
               XCAFDoc_ColorGen);
  auto wheelAxleProto = t_prototype();
  wheelAxleProto.shape = BuildWheelAxle(wheelProto.shape, axleProto.shape, L);
  wheelAxleProto.label = ST->AddShape(wheelAxleProto.shape, true);

  TDataStd_Name::Set(wheelAxleProto.label, "wheel-axle");

  auto chassisProto = t_prototype();
  chassisProto.shape = BuildChassis(wheelAxleProto.shape, CL);
  chassisProto.label = ST->AddShape(chassisProto.shape, true);

  TDataStd_Name::Set(chassisProto.label, "chassis");

  auto allWheelFaces = TopTools_IndexedMapOfShape();
  TopExp::MapShapes(wheelProto.shape, TopAbs_FACE, allWheelFaces);
  // the 2 is a bit too magic in my opinion. he selected the face in an editor
  // to find the index
  wheelProto.frontFace = TopoDS::Face(allWheelFaces(2));
  wheelProto.frontFaceLabel =
      ST->AddSubShape(wheelProto.label, wheelProto.frontFace);

  CT->SetColor(wheelProto.frontFaceLabel,
               Quantity_Color(0, 0, 1, Quantity_TOC_RGB), XCAFDoc_ColorSurf);

  auto status = app->SaveAs(doc, "doc.xbf");
  if (!(PCDM_SS_OK == status)) {
    return 1;
  }

  WriteStep(doc, "o.stp");
  return 0;
}
