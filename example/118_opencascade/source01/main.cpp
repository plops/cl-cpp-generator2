#include <GL/glut.h>
#include <memory>
#include <opencascade/BRepPrimAPI_MakeCylinder.hxx>
#include <opencascade/BinXCAFDrivers.hxx>
#include <opencascade/TDocStd_Application.hxx>
#include <opencascade/XCAFDoc_ColorTool.hxx>
#include <opencascade/XCAFDoc_DocumentTool.hxx>
#include <opencascade/XCAFDoc_ShapeTool.hxx>
#include <vector>

TopoDS_Shape BuildWheel(const double OD, const double W) {
  return BRepPrimAPI_MakeCylinder(gp_Ax2(gp::Origin(), gp::DX()), ((OD) / (2)),
                                  W);
}

TopoDS_Shape BuildAxle(const double D, const double L) {
  return BRepPrimAPI_MakeCylinder(gp_Ax2(gp::Origin(), gp::DX()), ((D) / (2)),
                                  L);
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
  auto wheelProto = t_prototype();
  wheelProto.shape = BuildWheel(OD, W);

  return 0;
}
