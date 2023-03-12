#include <GL/glut.h>
#include <memory>
#include <opencascade/BRepPrimAPI_MakeCylinder.hxx>
#include <opencascade/BinXCAFDrivers.hxx>
#include <opencascade/STEPCAFControl_Writer.hxx>
#include <opencascade/TDocStd_Application.hxx>
#include <opencascade/XCAFDoc_ColorTool.hxx>
#include <opencascade/XCAFDoc_DocumentTool.hxx>
#include <opencascade/XCAFDoc_ShapeTool.hxx>
#include <vector>

TopoDS_Shape BuildWheel(const double OD, const double W) {
  return BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(-W / 2, 0, 0), gp::DX()),
                                  ((OD) / (2)), W);
}

TopoDS_Shape BuildAxle(const double D, const double L) {
  return BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(-L / 2, 0, 0), gp::DX()),
                                  ((D) / (2)), L);
}

TopoDS_Shape BuildWheelAxle(const TopoDS_Shape &wheel, const TopoDS_Shape &axle,
                            const double L) {
  auto comp = TopoDS_Compound();
  auto bbuilder = BRep_Builder();
  auto wheelT_right = gp_Trsf();
  auto wheelT_left = gp_Trsf();
  wheelT_right.SetTranslationPart(gp_Vec(L / 2, 0, 0));
  wheelT_left.SetTranslationPart(gp_Vec(-L / 2, 0, 0));
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
  auto wheelProto = t_prototype();
  wheelProto.shape = BuildWheel(OD, W);
  wheelProto.label = ST->AddShape(wheelProto.shape, false);

  auto axleProto = t_prototype();
  axleProto.shape = BuildAxle(D, L);
  axleProto.label = ST->AddShape(axleProto.shape, false);

  auto wheelAxleProto = t_prototype();
  wheelAxleProto.shape = BuildWheelAxle(wheelProto.shape, axleProto.shape, L);
  wheelAxleProto.label = ST->AddShape(wheelAxleProto.shape, true);

  auto chassisProto = t_prototype();
  chassisProto.shape = BuildChassis(wheelAxleProto.shape, CL);
  chassisProto.label = ST->AddShape(chassisProto.shape, true);

  auto status = app->SaveAs(doc, "doc.xbf");
  if (!(PCDM_SS_OK == status)) {
    return 1;
  }

  WriteStep(doc, "o.stp");
  return 0;
}
