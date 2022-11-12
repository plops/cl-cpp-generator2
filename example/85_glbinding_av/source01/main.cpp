#include <cassert>
#include <chrono>
#include <glbinding/AbstractFunction.h>
#include <glbinding/CallbackMask.h>
#include <glbinding/FunctionCall.h>
#include <glbinding/gl32core/gl.h>
#include <glbinding/glbinding.h>
#include <iomanip>
#include <iostream>
#include <thread>
using namespace gl32core;
using namespace glbinding;
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h>
#include <imgui.h>
#define GLFW_INCLUDE_NONE
#include "GlfwWindow.h"
#include "ImguiHandler.h"
#include "Video.h"
#include <GLFW/glfw3.h>
#include <avcpp/av.h>
#include <avcpp/codec.h>
#include <avcpp/codeccontext.h>
#include <avcpp/ffmpeg.h>
#include <avcpp/formatcontext.h>
#include <cxxopts.hpp>
const std::chrono::time_point<std::chrono::high_resolution_clock> g_start_time =
    std::chrono::high_resolution_clock::now();
void lprint(std::initializer_list<std::string> il, std::string file, int line,
            std::string fun) {
  std::chrono::duration<double> timestamp(0);
  timestamp = ((std::chrono::high_resolution_clock::now()) - (g_start_time));
  const auto defaultWidth = 10;
  (std::cout) << (std::setw(defaultWidth)) << (timestamp.count()) << (" ")
              << (file) << (":") << (std::to_string(line)) << (" ") << (fun)
              << (" ") << (std::this_thread::get_id()) << (" ");
  for (const auto &elem : il) {
    (std::cout) << (elem);
  }
  (std::cout) << (std::endl) << (std::flush);
}
int main(int argc, char **argv) {
  lprint({"start", " ", " argc='", std::to_string(argc), "'"}, __FILE__,
         __LINE__, &(__PRETTY_FUNCTION__[0]));
  auto options = cxxopts::Options("gl-video-viewer", "play videos with opengl");
  auto positional = std::vector<std::string>();
  (((options.add_options())("h,help", "Print usage"))(
      "i,internal-tex-format", "data format of texture",
      cxxopts::value<int>()->default_value("3")))(
      "filenames", "The filenames of videos to display",
      cxxopts::value<std::vector<std::string>>(positional));
  options.parse_positional({"filenames"});
  auto opt_res = options.parse(argc, argv);
  if (opt_res.count("help")) {
    (std::cout) << (options.help()) << (std::endl);
    exit(0);
  }
  auto texFormatIdx = opt_res["internal-tex-format"].as<int>();
  assert((0) <= (texFormatIdx));
  assert((texFormatIdx) < (8));
  auto texFormats = std::array<gl::GLenum, 8>(
      {GL_RGBA, GLenum::GL_RGB8, GLenum::GL_R3_G3_B2, GLenum::GL_RGBA2,
       GLenum::GL_RGB9_E5, GLenum::GL_SRGB8, GLenum::GL_RGB8UI,
       GLenum::GL_COMPRESSED_RGB});
  auto texFormat = texFormats[texFormatIdx];
  auto win = GlfwWindow();
  auto width = int(0);
  auto height = int(0);
  lprint({"initialize glbinding", " "}, __FILE__, __LINE__,
         &(__PRETTY_FUNCTION__[0]));
  // if second arg is false: lazy function pointer loading
  glbinding::initialize(glfwGetProcAddress, false);
  {
    const float r = (0.40f);
    const float g = (0.40f);
    const float b = (0.20f);
    const float a = (1.0f);
    glClearColor(r, g, b, a);
  }
  auto imguiHandler = ImguiHandler(win.GetWindow());
  av::init();
  auto video = Video(positional.at(0));
  const auto radius = (10.f);
  bool video_is_initialized_p = false;
  int image_width = 0;
  int image_height = 0;
  GLuint image_texture = 0;
  lprint({"start loop", " "}, __FILE__, __LINE__, &(__PRETTY_FUNCTION__[0]));
  while (!(win.WindowShouldClose())) {
    glfwPollEvents();
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    auto showDemoWindow = true;
    ImGui::ShowDemoWindow(&showDemoWindow);
    ([&width, &height, win]() {
      // react to changing window size
      auto oldwidth = width;
      auto oldheight = height;
      auto [nwidth, nheight] = win.GetWindowSize();
      width = nwidth;
      height = nheight;
      if ((((width) != (oldwidth)) || ((height) != (oldheight)))) {
        lprint({"window size has changed", " ", " width='",
                std::to_string(width), "'", " height='", std::to_string(height),
                "'"},
               __FILE__, __LINE__, &(__PRETTY_FUNCTION__[0]));
        glViewport(0, 0, width, height);
      }
    })();
    {
      av::Packet pkt;
      while (pkt = video.readPacket()) {
        if (!((video.videoStream) == (pkt.streamIndex()))) {
          continue;
        }
        auto ts = pkt.ts();
        auto frame = video.decode();
        ts = frame.pts();
        if (((frame.isComplete()) && (frame.isValid()))) {
          auto *data = frame.data(0);
          image_width = frame.raw()->linesize[0];
          image_height = frame.height();
          auto init_width = image_width;
          auto init_height = image_height;
          if (!video_is_initialized_p) {
            // initialize texture for video frames
            glGenTextures(1, &image_texture);
            glBindTexture(GL_TEXTURE_2D, image_texture);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            lprint({"prepare texture", " ", " init_width='",
                    std::to_string(init_width), "'", " image_width='",
                    std::to_string(image_width), "'", " image_height='",
                    std::to_string(image_height), "'", " frame.width()='",
                    std::to_string(frame.width()), "'"},
                   __FILE__, __LINE__, &(__PRETTY_FUNCTION__[0]));
            glTexImage2D(GL_TEXTURE_2D, 0, texFormat, image_width, image_height,
                         0, GLenum::GL_LUMINANCE, GL_UNSIGNED_BYTE, nullptr);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, image_width, image_height,
                            GLenum::GL_LUMINANCE, GL_UNSIGNED_BYTE, data);
            video_is_initialized_p = true;
          } else {
            // update texture with new frame
            glBindTexture(GL_TEXTURE_2D, image_texture);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, image_width, image_height,
                            GLenum::GL_LUMINANCE, GL_UNSIGNED_BYTE, data);
          }
          break;
        }
      }
      // draw frame
      ImGui::Begin("video texture");
      ImGui::Text("width = %d", image_width);
      ImGui::Image(
          reinterpret_cast<void *>(static_cast<intptr_t>(image_texture)),
          ImVec2(static_cast<float>(image_width),
                 static_cast<float>(image_height)));
      auto val_old = static_cast<float>(pkt.ts().seconds());
      auto val = val_old;
      ImGui::SliderFloat("time", &val, video.startTime(), video.duration(),
                         "%.3f");
      if (!((val) == (val_old))) {
        // perform seek operation
        video.seek(val);
      }
      ImGui::End();
      ImGui::Render();
      glClear(GL_COLOR_BUFFER_BIT);
      ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
      win.SwapBuffers();
    }
  }
  return 0;
}