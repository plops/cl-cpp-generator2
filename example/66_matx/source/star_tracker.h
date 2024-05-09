#include <chrono>
#include <fstream>
#include <iostream>
#include <thread>
;
#include <filament/Camera.h>
#include <filament/Engine.h>
#include <filament/IndexBuffer.h>
#include <filament/Material.h>
#include <filament/MaterialInstance.h>
#include <filament/RenderableManager.h>
#include <filament/Scene.h>
#include <filament/Skybox.h>
#include <filament/TransformManager.h>
#include <filament/VertexBuffer.h>
#include <filament/View.h>
#include <filamentapp/Config.h>
#include <filamentapp/FilamentApp.h>
#include <utils/EntityManager.h>
;
#include "generated/resources.h"
;
class App {
public:
  filament::VertexBuffer *vb;
  filament::IndexBuffer *ib;
  filament::Material *mat;
  filament::Camera *cam;
  utils::Entity camera;
  filament::Skybox *skybox;
  utils::Entity renderable;
};
class Vertex {
public:
  filament::math::float2 position;
  uint32_t color;
};
class StarEntry {
public:
  float magnitude, ra, dec;
};
