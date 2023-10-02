#include "Physics.h"
#include <SDL.h>
#include <SDL_opengl.h>
#include <imgui.h>
#include <imgui_impl_opengl3.h>
#include <imgui_impl_sdl2.h>
#include <iostream>
#include <memory>

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
  auto physics = std::make_unique<Physics>();
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
      auto px = 0.F;
      auto py = 0.F;
      auto angle = 0.F;
      std::tie(px, py, angle) = physics->Step();
      auto draw = ImGui::GetBackgroundDrawList();
      auto rad = 1.00e+2F;
      auto ppx = 100 * px;
      auto ppy = 100 * py;
      auto sx = sin(angle);
      auto sy = cos(angle);
      draw->AddLine(ImVec2(ppx, ppy), ImVec2(ppx + rad * sx, ppy + rad * sy),
                    ImGui::GetColorU32(ImGuiCol_Button), 4.0F);
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
