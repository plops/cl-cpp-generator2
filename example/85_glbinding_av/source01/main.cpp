#include <cassert>
#include <chrono>
#include <glbinding/gl32core/gl.h>
#include <glbinding/glbinding.h>
#include <iomanip>
#include <iostream>
#include <thread>
using namespace gl32core;
using namespace glbinding;
#include "GlfwWindow.h"
#include "ImguiHandler.h"
#include "Texture.h"
#include "Video.h"
#include <avcpp/av.h>
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
  lprint({"initialize glbinding", " "}, __FILE__, __LINE__,
         &(__PRETTY_FUNCTION__[0]));
  // if second arg is false: lazy function pointer loading
  glbinding::initialize(win.GetProcAddress, false);
  {
    const float r = (0.40f);
    const float g = (0.40f);
    const float b = (0.20f);
    const float a = (1.0f);
    glClearColor(r, g, b, a);
  }
  auto imgui = ImguiHandler(win.GetWindow());
  av::init();
  auto video = Video(positional.at(0));
  lprint({"start loop", " "}, __FILE__, __LINE__, &(__PRETTY_FUNCTION__[0]));
  while (!(win.WindowShouldClose())) {
    win.PollEvents();
    imgui.NewFrame();
    {
      static int oldwidth = 0;
      static int oldheight = 0;
      // react to changing window size
      auto [width, height] = win.GetWindowSize();
      if ((((width) != (oldwidth)) || ((height) != (oldheight)))) {
        lprint({"window size has changed", " ", " width='",
                std::to_string(width), "'", " height='", std::to_string(height),
                "'"},
               __FILE__, __LINE__, &(__PRETTY_FUNCTION__[0]));
        glViewport(0, 0, width, height);
        oldwidth = width;
        oldheight = height;
      }
    }
    {
      av::Packet pkt;
      auto texture = Texture(640, 480, static_cast<unsigned int>(texFormat));
      while (pkt = video.readPacket()) {
        if (!((video.videoStream) == (pkt.streamIndex()))) {
          continue;
        }
        auto ts = pkt.ts();
        auto frame = video.decode();
        ts = frame.pts();
        if (((frame.isComplete()) && (frame.isValid()))) {
          auto *data = frame.data(0);
          auto w = frame.raw()->linesize[0];
          auto h = frame.height();
          texture.Reset(data, w, h, static_cast<unsigned int>(texFormat));
          break;
        }
      }
      // draw frame
      imgui.Begin("video texture");
      imgui.Image(texture.GetImageTexture(), texture.GetWidth(),
                  texture.GetHeight());
      auto val_old = static_cast<float>(pkt.ts().seconds());
      auto val = val_old;
      imgui.SliderFloat("time", &val, video.startTime(), video.duration(),
                        "%.3f");
      if (!((val) == (val_old))) {
        // perform seek operation
        video.seek(val);
      }
      imgui.End();
      imgui.Render();
      glClear(GL_COLOR_BUFFER_BIT);
      imgui.RenderDrawData();
      win.SwapBuffers();
    }
  }
  return 0;
}