#include <SDL.h>
#include <SDL_opengl.h>
#include <box2d/box2d.h>
#include <imgui.h>
#include <imgui_impl_opengl3.h>
#include <imgui_impl_sdl2.h>
#include <iostream>

int main(int argc, char **argv) {
  if (0 != SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER)) {
    std::cout << "Error"
              << " SDL_GetError()='" << SDL_GetError() << "' " << std::endl;
    return -1;
  }
  auto glsl_version = "#version 130";
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, 0);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);
  SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
  SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
  SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);
  SDL_SetHint(SDL_HINT_IME_SHOW_UI, "1");
  auto *window = SDL_CreateWindow(
      "imgui_sdl2_bullet_gears_designer", SDL_WINDOWPOS_CENTERED,
      SDL_WINDOWPOS_CENTERED, 1280, 720,
      SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI);
  if (!window) {
    throw std::runtime_error("Error creating GL window");
  }
  auto gl_context = SDL_GL_CreateContext(window);
  SDL_GL_MakeCurrent(window, gl_context);
  SDL_GL_SetSwapInterval(1);
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  auto *io = &ImGui::GetIO();
  io->ConfigFlags = io->ConfigFlags | ImGuiConfigFlags_NavEnableKeyboard;
  ImGui::StyleColorsDark();
  ImGui_ImplSDL2_InitForOpenGL(window, gl_context);
  ImGui_ImplOpenGL3_Init(glsl_version);
  glEnable(GL_CULL_FACE);
  auto physics = [&]() -> std::tuple<float, float, float> {
    // https://github.com/erincatto/box2d/blob/main/unit-test/hello_world.cpp

    const auto timeStep = 1.0F / 60.F;
    const auto velocityIterations = 6;
    const auto positionIterations = 2;
    static auto is_initialized = false;
    static auto world = b2World(b2Vec2(0.F, -10.F));
    b2Body *body = nullptr;
    if (!is_initialized) {
      auto groundBodyDef = b2BodyDef();
      groundBodyDef.position.Set(0.F, -10.F);
      auto groundBody = world.CreateBody(&groundBodyDef);
      auto groundBox = b2PolygonShape();
      groundBox.SetAsBox(50.F, 10.F);
      groundBody->CreateFixture(&groundBox, 0.F);
      auto bodyDef = b2BodyDef();
      bodyDef.type = b2_dynamicBody;
      bodyDef.position.Set(0.F, 4.0F);
      body = world.CreateBody(&bodyDef);
      auto dynamicBox = b2PolygonShape();
      dynamicBox.SetAsBox(1.0F, 1.0F);
      auto fixtureDef = b2FixtureDef();
      fixtureDef.shape = &dynamicBox;
      fixtureDef.density = 1.0F;
      fixtureDef.friction = 0.30F;
      body->CreateFixture(&fixtureDef);
      is_initialized = true;
    }
    world.Step(timeStep, velocityIterations, positionIterations);
    auto position = body->GetPosition();
    auto angle = body->GetAngle();
    std::cout << ""
              << " position.x='" << position.x << "' "
              << " position.y='" << position.y << "' "
              << " angle='" << angle << "' " << std::endl;
    return std::make_tuple(position.x, position.y, angle);
  };
  auto done = false;
  auto handle_events = [&](auto *window_, auto *done_) {
    auto event = SDL_Event();
    while (SDL_PollEvent(&event)) {
      ImGui_ImplSDL2_ProcessEvent(&event);
      if (SDL_QUIT == event.type) {
        *done_ = true;
      }
      if (SDL_WINDOWEVENT == event.type &&
          SDL_WINDOWEVENT_CLOSE == event.window.event &&
          event.window.windowID == SDL_GetWindowID(window_)) {
        *done_ = true;
      }
    }
  };
  auto new_frame = [&]() {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplSDL2_NewFrame();
    ImGui::NewFrame();
  };
  auto demo_window = [&]() {
    static bool show_demo = true;
    if (show_demo) {
      ImGui::ShowDemoWindow(&show_demo);
    }
  };
  auto swap = [&]() {
    ImGui::Render();
    glViewport(0, 0, static_cast<int>(io->DisplaySize.x),
               static_cast<int>(io->DisplaySize.y));
    glClearColor(0.F, 0.F, 0.F, 1.0F);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    SDL_GL_SwapWindow(window);
  };
  try {
    while (!done) {
      handle_events(window, &done);
      physics();
      new_frame();
      demo_window();
      swap();
    }
  } catch (const std::runtime_error &e) {
    std::cout << "error"
              << " e.what()='" << e.what() << "' " << std::endl;
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    ImGui::DestroyContext();
    SDL_GL_DeleteContext(gl_context);
    SDL_Quit();
    return 1;
  }
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplSDL2_Shutdown();
  ImGui::DestroyContext();
  SDL_GL_DeleteContext(gl_context);
  SDL_Quit();
  return 0;
}
