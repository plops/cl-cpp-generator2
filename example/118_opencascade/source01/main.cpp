#include <GL/glut.h>
#include <memory>
#include <opencascade/BinXCAFDrivers.hxx>
#include <opencascade/TDocStd_Application.hxx>
#include <vector>

int main(int argc, char **argv) {
  auto app = opencascade::handle<TDocStd_Application>(new TDocStd_Application);
  BinXCAFDrivers::DefineFormat(app);
  auto doc = opencascade::handle<TDocStd_Document>();
  app->NewDocument("BinXCAF", doc);

  return 0;
}
