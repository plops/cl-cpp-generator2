
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
  auto engine = filament::Engine::create();
  engine->destroy(&engine);
  return 0;
};