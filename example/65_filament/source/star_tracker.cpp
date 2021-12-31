#include "star_tracker.h"
#include <imgui.h>
using namespace filament;
using utils::Entity;
using utils::EntityManager;
int main(int argc, char **argv) {
  auto file = std::ifstream("/home/martin/stage/cl-cpp-generator2/example/"
                            "65_filament/script/out_118218x3_float32.raw",
                            ((std::ios::in) | (std::ios::binary)));
  auto size = 118218;
  std::vector<StarEntry> star_data;
  star_data.resize(size);
  file.read((char *)star_data.data(), ((size) * (sizeof(StarEntry))));
  file.close();
  static Vertex triangle_vertices[118218];
  static uint16_t triangle_indices[118218];
  for (auto i = 0; (i) < (size); (i) += (1)) {
    triangle_indices[i] = i;
  }
  for (auto i = 0; (i) < (size); (i) += (1)) {
    triangle_vertices[i].position.x =
        ((1) * (((((star_data[i].ra) - (180))) / (180))));
    triangle_vertices[i].position.y = ((2) * (((star_data[i].dec) / (90))));
    auto m = static_cast<int>(
        ((((256) / (16))) * (((16) - (((2) + (star_data[i].magnitude)))))));
    triangle_vertices[i].color =
        (((255) << (((3) * (8)))) | ((255) << (((2) * (8)))) |
         ((m) << (((1) * (8)))) | ((255) << (((0) * (8)))));
  }
  Config config;
  App app;
  config.title = "hello triangle";
  config.backend = Engine::Backend::VULKAN;
  auto setup = [&app](Engine *engine, View *view, Scene *scene) {
    app.skybox = Skybox::Builder()
                     .color({(0.10f), (0.1250f), (0.250f), (1.0f)})
                     .build(*engine);
    scene->setSkybox(app.skybox);
    view->setPostProcessingEnabled(false);
    static_assert((12) == (sizeof(Vertex)), "strange vertex size");
    app.vb = VertexBuffer::Builder()
                 .vertexCount(118218)
                 .bufferCount(1)
                 .attribute(VertexAttribute::POSITION, 0,
                            VertexBuffer::AttributeType::FLOAT2, 0, 12)
                 .attribute(VertexAttribute::COLOR, 0,
                            VertexBuffer::AttributeType::UBYTE4, 8, 12)
                 .normalized(VertexAttribute::COLOR)
                 .build(*engine);
    app.vb->setBufferAt(
        *engine, 0,
        VertexBuffer::BufferDescriptor(
            triangle_vertices, ((sizeof(float)) * (2) * (118218)), nullptr));
    app.ib = IndexBuffer::Builder()
                 .indexCount(118218)
                 .bufferType(IndexBuffer::IndexType::USHORT)
                 .build(*engine);
    app.ib->setBuffer(*engine, IndexBuffer::BufferDescriptor(
                                   triangle_indices,
                                   ((sizeof(uint16_t)) * (118218)), nullptr));
    app.mat = Material::Builder()
                  .package(RESOURCES_BAKEDCOLOR_DATA, RESOURCES_BAKEDCOLOR_SIZE)
                  .build(*engine);
    app.renderable = EntityManager::get().create();
    RenderableManager::Builder(1)
        .boundingBox({{-1, -1, -1}, {1, 1, 1}})
        .material(0, app.mat->getDefaultInstance())
        .geometry(0, RenderableManager::PrimitiveType::POINTS, app.vb, app.ib,
                  0, 118218)
        .culling(false)
        .receiveShadows(false)
        .castShadows(false)
        .build(*engine, app.renderable);
    scene->addEntity(app.renderable);
    app.camera = utils::EntityManager::get().create();
    app.cam = engine->createCamera(app.camera);
    view->setCamera(app.cam);
  };
  auto cleanup = [&app](Engine *engine, View *view, Scene *scene) {
    engine->destroy(app.skybox);
    engine->destroy(app.renderable);
    engine->destroy(app.mat);
    engine->destroy(app.vb);
    engine->destroy(app.ib);
    engine->destroyCameraComponent(app.camera);
    utils::EntityManager::get().destroy(app.camera);
  };
  auto gui = [&app](Engine *engine, View *view) {
    ImGui::Begin("Parameters");
    {
      static float gain = (1.0f);
      ImGui::SliderFloat("gain", &gain, (0.f), (1.0f));
    }
    ImGui::End();
  };
  static FilamentApp &filament_app = FilamentApp::get();
  filament_app.animate([&app](Engine *engine, View *view, double now) {
    auto zoom = 1.5f;
    auto w = view->getViewport().width;
    auto h = view->getViewport().height;
    auto aspect = ((static_cast<float>(w)) / (h));
    app.cam->setProjection(Camera::Projection::ORTHO,
                           ((-1) * (aspect) * (zoom)), ((aspect) * (zoom)),
                           ((-1) * (zoom)), zoom, 0, 1);
    auto &tcm = engine->getTransformManager();
    tcm.setTransform(
        tcm.getInstance(app.renderable),
        filament::math::mat4f::rotation(now, filament::math::float3{0, 0, -1}));
  });
  filament_app.run(config, setup, cleanup, gui);
  return 0;
}