
#include "utils.h"

#include "globals.h"

#include "proto2.h"
;
#include <cstdlib>
#include <filament/IndexBuffer.h>
#include <filament/RenderableManager.h>
#include <filament/Scene.h>
#include <filament/VertexBuffer.h>
#include <filament/View.h>
#include <filamentapp/FilamentApp.h>
#include <utils/EntityManager.h>

State state = {};
using namespace std::chrono_literals;
struct App {
  filament::VertexBuffer *vb;
  filament::IndexBuffer *ib;
  filament::Camera *cam;
  utils::Entity renderable;
};
typedef struct App App;
int main() {
  Config config;
  config.title = "viewtest";
  config.backend = filament::Engine::Backend::VULKAN;
  App app;
  auto setup = [&app](filament::Engine *engine, filament::View *view,
                      filament::Scene *scene) {
    view->setClearColor({0, 0, 1, 1});
  };
  auto cleanup = [&app](filament::Engine *engine, filament::View *view,
                        filament::Scene *scene) {
    engine->destroy(app.renderable);
    engine->destroy(app.vb);
    engine->destroy(app.ib);
    engine->destroy(app.cam);
  };
  FilamentApp::get().run(config, setup, cleanup);
  return 0;
};