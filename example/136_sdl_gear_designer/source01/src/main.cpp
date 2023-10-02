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
  auto gravity = b2Vec2(0.F, -10.F);
  auto world = b2World(gravity);
  auto groundBodyDef = b2BodyDef();
  groundBodyDef.position.Set(0.F, -10.F);
  auto groundBody = world.CreateBody(&groundBodyDef);
  auto groundBox = b2PolygonShape();
  groundBox.SetAsBox(50.F, 10.F);
  groundBody->CreateFixture(&groundBox, 0.F);
  auto bodyDef = b2BodyDef();
  bodyDef.type = b2_dynamicBody;
  bodyDef.position.Set(0.F, 4.0F);
  auto body = world.CreateBody(&bodyDef);
  auto done = false;
  while (!done) {
    auto event = SDL_Event();
    while (SDL_PollEvent(&event)) {
      ImGui_ImplSDL2_ProcessEvent(&event);
      if (SDL_QUIT == event.type) {
        done = true;
      }
      if (SDL_WINDOWEVENT == event.type &&
          SDL_WINDOWEVENT_CLOSE == event.window.event &&
          event.window.windowID == SDL_GetWindowID(window)) {
        done = true;
      }
    }
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplSDL2_NewFrame();
    ImGui::NewFrame();
    static bool show_demo = true;
    if (show_demo) {
      ImGui::ShowDemoWindow(&show_demo);
    }
    ImGui::Render();
    glViewport(0, 0, static_cast<int>(io->DisplaySize.x),
               static_cast<int>(io->DisplaySize.y));
    glClearColor(0.F, 0.F, 0.F, 1.0F);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    SDL_GL_SwapWindow(window);
  }
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplSDL2_Shutdown();
  ImGui::DestroyContext();
  SDL_GL_DeleteContext(gl_context);
  SDL_Quit();
  return 0;
}
