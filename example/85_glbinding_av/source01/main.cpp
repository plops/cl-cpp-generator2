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
#include <fmt/core.h>
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
  const auto numTexFormats = 8;
  assert((0) <= (texFormatIdx));
  assert((texFormatIdx) < (numTexFormats));
  auto texFormatsString = std::array<std::string, numTexFormats>(
      {"GL_RGBA", "GLenum--GL_RGB8", "GLenum--GL_R3_G3_B2", "GLenum--GL_RGBA2",
       "GLenum--GL_RGB9_E5", "GLenum--GL_SRGB8", "GLenum--GL_RGB8UI",
       "GLenum--GL_COMPRESSED_RGB"});
  auto texFormats = std::array<gl::GLenum, numTexFormats>(
      {GL_RGBA, GLenum::GL_RGB8, GLenum::GL_R3_G3_B2, GLenum::GL_RGBA2,
       GLenum::GL_RGB9_E5, GLenum::GL_SRGB8, GLenum::GL_RGB8UI,
       GLenum::GL_COMPRESSED_RGB});
  auto texFormat = texFormats.at(texFormatIdx);
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
  auto video = std::make_unique<Video>(fn);
  const auto initW = 640;
  const auto initH = 480;
  auto texture = Texture(initW, initH, static_cast<int>(texFormat));
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
      while (([&pkt, &video]() -> bool {
        if (!(video->GetSuccess())) {
          return false;
        }
        pkt = video->readPacket();
        if ((pkt.size()) <= (0)) {
          return false;
        }
        if (!(pkt.flags())) {
          lprint({"normal pkt", " ", " pkt.size()='",
                  std::to_string(pkt.size()), "'", " pkt.flags()='",
                  std::to_string(pkt.flags()), "'"},
                 __FILE__, __LINE__, &(__PRETTY_FUNCTION__[0]));
        }
        if (((1) & (pkt.flags()))) {
          lprint({"pkt contains keyframe", " ", " pkt.size()='",
                  std::to_string(pkt.size()), "'", " pkt.flags()='",
                  std::to_string(pkt.flags()), "'"},
                 __FILE__, __LINE__, &(__PRETTY_FUNCTION__[0]));
        }
        if (((2) & (pkt.flags()))) {
          lprint({"pkt corrupt", " ", " pkt.size()='",
                  std::to_string(pkt.size()), "'", " pkt.flags()='",
                  std::to_string(pkt.flags()), "'"},
                 __FILE__, __LINE__, &(__PRETTY_FUNCTION__[0]));
          return false;
        }
        if (((4) & (pkt.flags()))) {
          lprint({"pkt discard", " ", " pkt.size()='",
                  std::to_string(pkt.size()), "'", " pkt.flags()='",
                  std::to_string(pkt.flags()), "'"},
                 __FILE__, __LINE__, &(__PRETTY_FUNCTION__[0]));
        }
        return true;
      })()) {
        if (!((video->videoStream) == (pkt.streamIndex()))) {
          continue;
        }
        auto ts = pkt.ts();
        auto frame = video->decode();
        ts = frame.pts();
        if (((frame.isComplete()) && (frame.isValid()))) {
          auto *data(frame.data(0));
          auto w = frame.raw()->linesize[0];
          auto h = frame.height();
          texture.Reset(data, w, h, static_cast<int>(texFormat));
          break;
        }
      }
      // draw frame
      imgui.Begin("video texture");
      ImGui::Text("fn = %s", fn.c_str());
      if (video->GetSuccess()) {
        imgui.Image(texture.GetImageTexture(), texture.GetWidth(),
                    texture.GetHeight());
        if (video->Seekable_p()) {
          auto val_old = static_cast<float>(pkt.ts().seconds());
          auto val = val_old;
          imgui.SliderFloat("time", &val, video->startTime(), video->duration(),
                            "%.3f");
          if (!((val) == (val_old))) {
            lprint({"perform seek operation", " "}, __FILE__, __LINE__,
                   &(__PRETTY_FUNCTION__[0]));
            video->seek(val);
          }
        } else {
          ImGui::Text("can't seek in file");
        }
      } else {
        ImGui::Text("could not open video file");
      }
      imgui.End();
      // window with file listing
      imgui.Begin("video files");
      static int item_current_idx = int(0);
      static int item_old_idx = int(0);
      const auto filesToShow = (40.f);
      ImGui::BeginListBox(
          "files", ImVec2(-FLT_MIN, ((filesToShow) *
                                     (ImGui::GetTextLineHeightWithSpacing()))));
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
            video = std::make_unique<Video>(fn);
          }
        }
        i = ((i) + (1));
      }
      ImGui::EndListBox();
      imgui.End();
      {
        // window with internal texture format listing
        imgui.Begin("internal texture format");
        static int fmt_current_idx = varInternalTextureFormat;
        static int fmt_old_idx = varInternalTextureFormat;
        const auto formatsToShow = (15.f);
        ImGui::BeginListBox(
            "files",
            ImVec2(-FLT_MIN, ((formatsToShow) *
                              (ImGui::GetTextLineHeightWithSpacing()))));
        auto i = 0;
        for (auto arg : texFormats) {
          auto selected_p = (i) == (fmt_current_idx);
          auto argString = texFormatsString.at(i);
          if (ImGui::Selectable(argString.c_str(), selected_p)) {
            fmt_current_idx = i;
          }
          if (selected_p) {
            ImGui::SetItemDefaultFocus();
            if (!((fmt_old_idx) == (fmt_current_idx))) {
              lprint({"change texture format", " ", " argString='", argString,
                      "'"},
                     __FILE__, __LINE__, &(__PRETTY_FUNCTION__[0]));
              fmt_old_idx = fmt_current_idx;
              varInternalTextureFormat = fmt_current_idx;
              texFormat = texFormats.at(varInternalTextureFormat);
            }
          }
          i = ((i) + (1));
        }
        ImGui::EndListBox();
        imgui.End();
      }
      imgui.Render();
      glClear(GL_COLOR_BUFFER_BIT);
      imgui.RenderDrawData();
      win.SwapBuffers();
    }
  }
  return 0;
}