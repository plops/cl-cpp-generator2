#include "GLProto.h"
#include <SDL.h>
#include <SDL_opengl.h>
#include <cmath>
#include <complex>
#include <format>
#include <imgui.h>
#include <imgui_impl_opengl3.h>
#include <imgui_impl_sdl2.h>
#include <iostream>
#include <memory>
#include <thread>
#include <unordered_map>
#include <vector>
class GuiException : public std::runtime_error {
public:
  using runtime_error::runtime_error;
};
auto slider_factory = []() {
  static auto values = std::unordered_map<std::string, float>();
  auto make_slider = [&](const std::string &label) {
    if (!values.contains(label)) {
      std::cout << std::format("make_slider init label='{}'\n", label);
      values[label] = 1.00e+2F;
    }
    return [label]() { return values[label]; };
  };
  auto draw_all_sliders = [&]() {
    ImGui::Begin("all-sliders");
    for (const auto &[key, value] : values) {
      ImGui::SliderFloat(key.c_str(), &values[key], 10.F, 6.00e+2F);
    }
    ImGui::End();
  };
  auto lookup_slider = [&](auto label) {
    if (values.contains(label)) {
      return values[label];
    }
    throw std::runtime_error(std::format("label '{}' undefined.", label));
    return 0.F;
  };
  return std::make_tuple(make_slider, draw_all_sliders, lookup_slider);
};

int main(int argc, char **argv) {
  std::cout << std::format("main entry point argc='{}' argv[0]='{}'\n", argc,
                           argv[0]);
  void *gl_context = nullptr;
  auto set_gl_attributes = [&]() {
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, 0);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK,
                        SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
    SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);
    SDL_SetHint(SDL_HINT_IME_SHOW_UI, "1");
  };
  auto init_gl = [&](auto &gl_context_) {
    if (0 != SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER)) {
      std::cout << std::format("Error SDL_GetError()='{}'\n", SDL_GetError());
      throw GuiException("Error in SDL_Init.");
    }
    set_gl_attributes();
    auto *window = SDL_CreateWindow(
        "imgui_sdl2_bullet_gears_designer", SDL_WINDOWPOS_CENTERED,
        SDL_WINDOWPOS_CENTERED, 1280, 720,
        SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI);
    if (!window) {
      throw GuiException("Error creating GL window");
    }
    gl_context_ = SDL_GL_CreateContext(window);
    SDL_GL_MakeCurrent(window, gl_context_);
    SDL_GL_SetSwapInterval(1);
    glEnable(GL_CULL_FACE);
    return window;
  };
  auto init_imgui = [&](auto window_, auto gl_context_) {
    std::cout << std::format("init_imgui\n");
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    auto *io = &ImGui::GetIO();
    io->ConfigFlags = io->ConfigFlags | ImGuiConfigFlags_NavEnableKeyboard;
    ImGui::StyleColorsDark();
    ImGui_ImplSDL2_InitForOpenGL(window_, gl_context_);
    auto glsl_version = "#version 130";
    ImGui_ImplOpenGL3_Init(glsl_version);
  };
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
  auto swap = [&](auto *window) {
    ImGui::Render();
    auto const *io = &ImGui::GetIO();
    glViewport(0, 0, static_cast<int>(io->DisplaySize.x),
               static_cast<int>(io->DisplaySize.y));
    glClearColor(0.F, 0.F, 0.F, 1.0F);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    SDL_GL_SwapWindow(window);
  };
  auto destroy_gl = [&](auto gl_context_) {
    std::cout << std::format("destroy_gl\n");
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    ImGui::DestroyContext();
    SDL_GL_DeleteContext(gl_context_);
    SDL_Quit();
  };
  try {
    auto *window = init_gl(gl_context);
    init_imgui(window, gl_context);
    auto service = GLProto::AsyncService();
    auto builder = grpc::ServerBuilder();
    builder.AddListeningPort("0.0.0.0:7777", grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    auto cq = builder.AddCompletionQueue();
    auto server = builder.BuildAndStart();
    server->Wait();
    auto [make_slider, draw_all_sliders, lookup_slider] = slider_factory();
    auto done = false;
    std::cout << std::format("start gui loop\n");
    while (!done) {
      handle_events(window, &done);
      new_frame();
      draw_all_sliders();
      demo_window();
      swap(window);
    }
  } catch (const std::runtime_error &e) {
    std::cout << std::format("error e.what()='{}'\n", e.what());
    destroy_gl(gl_context);
    return 1;
  }
  destroy_gl(gl_context);
  return 0;
}
