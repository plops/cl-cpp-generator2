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
  ((options.add_options())("h,help", "Print usage"))(
      "filenames", "The filenames of videos to display",
      cxxopts::value<std::vector<std::string>>(positional));
  options.parse_positional({"filenames"});
  auto opt_res = options.parse(argc, argv);
  if (opt_res.count("help")) {
    (std::cout) << (options.help()) << (std::endl);
    exit(0);
  }
  av::init();
  auto ctx = av::FormatContext();
  auto fn = positional.at(0);
  lprint({"open video file", " ", " fn='", fn, "'"}, __FILE__, __LINE__,
         &(__PRETTY_FUNCTION__[0]));
  ctx.openInput(fn);
  ctx.findStreamInfo();
  lprint({"stream info", " ", " ctx.seekable()='",
          std::to_string(ctx.seekable()), "'", " ctx.startTime().seconds()='",
          std::to_string(ctx.startTime().seconds()), "'",
          " ctx.duration().seconds()='",
          std::to_string(ctx.duration().seconds()), "'",
          " ctx.streamsCount()='", std::to_string(ctx.streamsCount()), "'"},
         __FILE__, __LINE__, &(__PRETTY_FUNCTION__[0]));
  ctx.seek({static_cast<long int>(
                floor(((100) * ((((0.50f)) * (ctx.duration().seconds())))))),
            {1, 100}});
  ssize_t videoStream = -1;
  av::Stream vst;
  std::error_code ec;
  for (auto i = 0; (i) < (ctx.streamsCount()); (i) += (1)) {
    auto st = ctx.stream(i);
    if ((AVMEDIA_TYPE_VIDEO) == (st.mediaType())) {
      videoStream = i;
      vst = st;
      break;
    }
  }
  if (vst.isNull()) {
    lprint({"Video stream not found", " "}, __FILE__, __LINE__,
           &(__PRETTY_FUNCTION__[0]));
  }
  av::VideoDecoderContext vdec;
  if (vst.isValid()) {
    vdec = av::VideoDecoderContext(vst);
    auto codec = av::findDecodingCodec(vdec.raw()->codec_id);
    vdec.setCodec(codec);
    vdec.setRefCountedFrames(true);
    vdec.open({{"threads", "1"}}, av::Codec(), ec);
    if (ec) {
      lprint({"can't open codec", " "}, __FILE__, __LINE__,
             &(__PRETTY_FUNCTION__[0]));
    }
  }
  auto *window = ([]() -> GLFWwindow * {
    lprint({"initialize GLFW3", " "}, __FILE__, __LINE__,
           &(__PRETTY_FUNCTION__[0]));
    if (!(glfwInit())) {
      lprint({"glfwInit failed", " "}, __FILE__, __LINE__,
             &(__PRETTY_FUNCTION__[0]));
    }
    glfwWindowHint(GLFW_VISIBLE, true);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, true);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    lprint({"create GLFW3 window", " "}, __FILE__, __LINE__,
           &(__PRETTY_FUNCTION__[0]));
    const auto startWidth = 800;
    const auto startHeight = 600;
    auto window =
        glfwCreateWindow(startWidth, startHeight, "glfw", nullptr, nullptr);
    if (!(window)) {
      lprint({"can't create glfw window", " "}, __FILE__, __LINE__,
             &(__PRETTY_FUNCTION__[0]));
    }
    lprint({"initialize GLFW3 context for window", " "}, __FILE__, __LINE__,
           &(__PRETTY_FUNCTION__[0]));
    glfwMakeContextCurrent(window);
    // configure Vsync, 1 locks to 60Hz, FIXME: i should really check glfw
    // errors
    glfwSwapInterval(0);
    return window;
  })();
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
  lprint({"initialize ImGui", " "}, __FILE__, __LINE__,
         &(__PRETTY_FUNCTION__[0]));
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  auto io = ImGui::GetIO();
  io.ConfigFlags = ((io.ConfigFlags) | (ImGuiConfigFlags_NavEnableKeyboard));
  ImGui::StyleColorsLight();
  {
    const auto installCallbacks = true;
    ImGui_ImplGlfw_InitForOpenGL(window, installCallbacks);
  }
  const auto glslVersion = "#version 150";
  ImGui_ImplOpenGL3_Init(glslVersion);
  const auto radius = (10.f);
  bool video_is_initialized_p = false;
  int image_width = 0;
  int image_height = 0;
  GLuint image_texture = 0;
  lprint({"start loop", " "}, __FILE__, __LINE__, &(__PRETTY_FUNCTION__[0]));
  while (!(glfwWindowShouldClose(window))) {
    glfwPollEvents();
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    {
      ImGui::NewFrame();
      auto showDemoWindow = true;
      ImGui::ShowDemoWindow(&showDemoWindow);
    }
    ([&width, &height, window]() {
      // react to changing window size
      auto oldwidth = width;
      auto oldheight = height;
      glfwGetWindowSize(window, &width, &height);
      if ((((width) != (oldwidth)) || ((height) != (oldheight)))) {
        lprint({"window size has changed", " ", " width='",
                std::to_string(width), "'", " height='", std::to_string(height),
                "'"},
               __FILE__, __LINE__, &(__PRETTY_FUNCTION__[0]));
        glViewport(0, 0, width, height);
      }
    })();
    {
      std::error_code ec;
      av::Packet pkt;
      while (pkt = ctx.readPacket(ec)) {
        if (ec) {
          lprint({"packet reading error", " ", " ec.message()='", ec.message(),
                  "'"},
                 __FILE__, __LINE__, &(__PRETTY_FUNCTION__[0]));
        }
        if (!((videoStream) == (pkt.streamIndex()))) {
          continue;
        }
        auto ts = pkt.ts();
        auto frame = vdec.decode(pkt, ec);
        if (ec) {
          lprint({"error", " ", " ec.message()='", ec.message(), "'"}, __FILE__,
                 __LINE__, &(__PRETTY_FUNCTION__[0]));
        }
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
            glTexImage2D(GL_TEXTURE_2D, 0, GLenum::GL_RGBA2, image_width,
                         image_height, 0, GLenum::GL_LUMINANCE,
                         GL_UNSIGNED_BYTE, nullptr);
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
      ImGui::Text("fn = %s", fn.c_str());
      ImGui::Image(
          reinterpret_cast<void *>(static_cast<intptr_t>(image_texture)),
          ImVec2(static_cast<float>(image_width),
                 static_cast<float>(image_height)));
      auto val_old = static_cast<float>(pkt.ts().seconds());
      auto val = val_old;
      ImGui::SliderFloat("time", &val,
                         static_cast<float>(ctx.startTime().seconds()),
                         static_cast<float>(ctx.duration().seconds()), "%.3f");
      if (!((val) == (val_old))) {
        // perform seek operation
        ctx.seek({static_cast<long int>(floor(((1000) * (val)))), {1, 1000}});
      }
      ImGui::End();
      ImGui::Render();
      glClear(GL_COLOR_BUFFER_BIT);
      ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
      glfwSwapBuffers(window);
    }
  }
  lprint({"Shutdown ImGui and GLFW3", " "}, __FILE__, __LINE__,
         &(__PRETTY_FUNCTION__[0]));
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
  glfwDestroyWindow(window);
  glfwTerminate();
  return 0;
}