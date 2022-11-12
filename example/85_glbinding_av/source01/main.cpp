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
#include "GlfwWindow.h"
#include "ImguiHandler.h"
#include "Texture.h"
#include "Video.h"
#include <avcpp/av.h>
#include <avcpp/formatcontext.h>
#include <imgui.h>
#include <popl.hpp>
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
  auto op = popl::OptionParser("allowed opitons");
  auto varInternalTextureFormat = int(3);
  auto helpOption = op.add<popl::Switch>("h", "help", "produce help message");
  auto verboseOption =
      op.add<popl::Switch>("v", "verbose", "produce verbose output");
  auto texformatOption = op.add<popl::Value<int>>(
      "T", "texformat", "choose internal texture format", 3,
      &varInternalTextureFormat);
  op.parse(argc, argv);
  if (helpOption->count()) {
    (std::cout) << (op) << (std::endl);
    exit(0);
  }
  auto texFormatIdx = varInternalTextureFormat;
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
  if (verboseOption->is_set()) {
    glbinding::setCallbackMask(
        ((CallbackMask::After) | (CallbackMask::ParametersAndReturnValue)));
    glbinding::setAfterCallback([](const glbinding::FunctionCall &call) {
      auto fun = call.function->name();
      lprint({"cb", " ", " fun='", fun, "'"}, __FILE__, __LINE__,
             &(__PRETTY_FUNCTION__[0]));
    });
  }
  {
    const float r = (0.40f);
    const float g = (0.40f);
    const float b = (0.20f);
    const float a = (1.0f);
    glClearColor(r, g, b, a);
  }
  auto imgui = ImguiHandler(win.GetWindow());
  av::init();

  auto fn = op.non_option_args().at(0);
  auto video = new Video(fn);
  auto texture = Texture(640, 480, static_cast<unsigned int>(texFormat));
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
      while (pkt = video->readPacket()) {
        if (!((video->videoStream) == (pkt.streamIndex()))) {
          continue;
        }
        auto ts = pkt.ts();
        auto frame = video->decode();
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
      ImGui::Text("fn = %s", video->fn.c_str());
      imgui.Image(texture.GetImageTexture(), texture.GetWidth(),
                  texture.GetHeight());
      auto val_old = static_cast<float>(pkt.ts().seconds());
      auto val = val_old;
      imgui.SliderFloat("time", &val, video->startTime(), video->duration(),
                        "%.3f");
      if (!((val) == (val_old))) {
        // perform seek operation
        video->seek(val);
      }
      imgui.End();
      imgui.Begin("video files");
      static int item_current_idx = int(0);
      static int item_old_idx = int(0);
      ImGui::BeginListBox("files");
      auto i = 0;
      for (auto arg : op.non_option_args()) {
        auto selected_p = (i) == (item_current_idx);
        if (ImGui::Selectable(arg.c_str(), selected_p)) {
          item_current_idx = i;
        }
        if (selected_p) {
          ImGui::SetItemDefaultFocus();
          if (!((item_old_idx) == (item_current_idx))) {
            lprint({"change video", " "}, __FILE__, __LINE__,
                   &(__PRETTY_FUNCTION__[0]));
            item_old_idx = item_current_idx;
            fn = arg;
            delete (video);
            video = new Video(fn);
          }
        }
        i = ((i) + (1));
      }
      ImGui::EndListBox();
      imgui.End();
      imgui.Render();
      glClear(GL_COLOR_BUFFER_BIT);
      imgui.RenderDrawData();
      win.SwapBuffers();
    }
  }
  return 0;
}