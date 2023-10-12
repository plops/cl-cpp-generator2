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
      std::cout << "make_slider init"
                << " label='" << label << "' " << std::endl;
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
  std::cout << "main entry point"
            << " argc='" << argc << "' "
            << " (argv)[(0)]='" << argv[0] << "' " << std::endl;
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
      std::cout << "Error"
                << " SDL_GetError()='" << SDL_GetError() << "' " << std::endl;
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
    std::cout << "init_imgui" << std::endl;
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
    std::cout << "destroy_gl" << std::endl;
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    ImGui::DestroyContext();
    SDL_GL_DeleteContext(gl_context_);
    SDL_Quit();
  };
  try {
    auto *window = init_gl(gl_context);
    init_imgui(window, gl_context);
    auto physics = std::make_unique<Physics>();
    auto [make_slider, draw_all_sliders, lookup_slider] = slider_factory();
    auto done = false;
    std::cout << "start gui loop" << std::endl;
    auto circle_factory = [&make_slider](auto count) {
      auto draw_circle = [&make_slider, count]() {
        auto draw = ImGui::GetBackgroundDrawList();
        auto radius = (make_slider(std::format("circle{}_radius", count)))();
        auto posx = (make_slider(std::format("circle{}_posx", count)))();
        auto posy = (make_slider(std::format("circle{}_posy", count)))();
        auto circum = 2 * std::numbers::pi_v<float> * radius;
        auto num_segments = std::max(7, static_cast<int>(ceil(circum / 5.0F)));
        draw->AddCircleFilled(ImVec2(posx, posy), radius,
                              ImGui::GetColorU32(ImGuiCol_Separator),
                              num_segments);
      };
      return draw_circle;
    };
    while (!done) {
      handle_events(window, &done);
      new_frame();
      draw_all_sliders();
      auto [px, py, angle] = physics->Step();
      auto draw = ImGui::GetBackgroundDrawList();
      auto rad = 1.00e+2F;
      auto ppx = 100 * (400 + px);
      auto ppy = 100 * (400 + py);
      auto sx = sin(angle);
      auto sy = cos(angle);
      draw->AddLine(ImVec2(ppx, ppy), ImVec2(ppx + rad * sx, ppy + rad * sy),
                    ImGui::GetColorU32(ImGuiCol_Text), 4.0F);
      (circle_factory(0))();
      (circle_factory(1))();
      auto posx0 = lookup_slider("circle0_posx");
      auto posy0 = lookup_slider("circle0_posy");
      auto radius0 = lookup_slider("circle0_radius");
      auto posx1 = lookup_slider("circle1_posx");
      auto posy1 = lookup_slider("circle1_posy");
      auto radius1 = lookup_slider("circle1_radius");
      auto imvec = [&](auto z) {
        return ImVec2(static_cast<float>(z.real()),
                      static_cast<float>(z.imag()));
      };
      auto draw_involute = [&](auto cx, auto cy, auto radius, auto tmax,
                               auto max_arc_step) {
        // https://mathworld.wolfram.com/CircleInvolute.html

        auto points = std::vector<ImVec2>();
        auto count = 0;
        auto dt = std::sqrt((2 * max_arc_step) / radius);
        auto tt = 0.;
        auto s_prev = 0.;
        while (tt <= tmax) {
          auto ds_dt = radius * tt;
          auto dt =
              0 < ds_dt ? (max_arc_step / ds_dt) : (2.00e-2F * max_arc_step);
          if (count < 4) {
            dt = 8.00e-2F;
          }
          auto circ = std::exp(std::complex<double>(0., tt));
          auto tang = std::complex<double>(circ.imag(), -1 * circ.real());
          auto s = 0.50 * radius * tt * tt;
          auto tangential_angle = tt;
          auto z = radius * (circ + tt * tang);
          points.emplace_back(imvec(std::complex<double>(cx, cy) + z));
          s_prev = s;
          tt += dt;
          count++;
        }
        auto draw = ImGui::GetBackgroundDrawList();
        draw->AddPolyline(points.data(), points.size(),
                          ImGui::GetColorU32(ImGuiCol_Text), ImDrawFlags_None,
                          3.0F);
      };
      draw_involute(static_cast<double>(posx0), static_cast<double>(posy0),
                    static_cast<double>(radius0), 0.50F * M_PI, 4.0);
      auto c1 = Circle({std::complex<double>(posx0, posy0), radius0});
      auto c2 = Circle({std::complex<double>(posx1, posy1), radius1});
      auto [z0, z1] = findInnerTangent(c1, c2);
      draw->AddLine(imvec(z0), imvec(z1), ImGui::GetColorU32(ImGuiCol_Text),
                    4.0F);
      auto [z00, z2] = findInnerTangent(c2, c1);
      draw->AddLine(imvec(z1), imvec(z2), ImGui::GetColorU32(ImGuiCol_Text),
                    4.0F);
      draw->AddLine(imvec(c1.center), imvec(c2.center),
                    ImGui::GetColorU32(ImGuiCol_Text), 2.0F);
      demo_window();
      swap(window);
    }
  } catch (const std::runtime_error &e) {
    std::cout << "error"
              << " e.what()='" << e.what() << "' " << std::endl;
    destroy_gl(gl_context);
    return 1;
  }
  destroy_gl(gl_context);
  return 0;
}
