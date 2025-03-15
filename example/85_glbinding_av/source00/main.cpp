#include <thread>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cassert> 
#include <glbinding/gl32core/gl.h>
#include <glbinding/glbinding.h>
#include <glbinding/CallbackMask.h>
#include <glbinding/FunctionCall.h>
#include <glbinding/AbstractFunction.h> 
using namespace gl32core; 
using namespace glbinding;  
#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_opengl3.h> 
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h> 
#include <avcpp/av.h>
#include <avcpp/ffmpeg.h> 
#include <avcpp/formatcontext.h>
#include <avcpp/codec.h>
#include <avcpp/codeccontext.h>  
#include <cxxopts.hpp>  
const std::chrono::time_point<std::chrono::high_resolution_clock> g_start_time = std::chrono::high_resolution_clock::now(); 
// lprint not needed 

int main (int argc, char** argv)        {
        spdlog::info("start  argc='{}'", argc);
            auto options {cxxopts::Options("gl-video-viewer", "play videos with opengl")}; 
        auto positional {std::vector<std::string>()}; 
    (((options.add_options())("h,help", "Print usage"))("i,internal-tex-format", "data format of texture", (cxxopts::value<int>())->(default_value("3"))))("filenames", "The filenames of videos to display", cxxopts::value<std::vector<std::string>>(positional));
    options.parse_positional({"filenames"}); 
        auto opt_res {options.parse(argc, argv)}; 
    if ( opt_res.count("help") ) {
                        (std::cout)<<(options.help())<<(std::endl);
        exit(0); 
} 
        auto texFormatIdx {(opt_res)[("internal-tex-format")].as<int>()}; 
    assert((0)<=(texFormatIdx));
    assert((texFormatIdx)<(8));
        auto texFormats {std::array<gl::GLenum,8>({GL_RGBA, GLenum::GL_RGB8, GLenum::GL_R3_G3_B2, GLenum::GL_RGBA2, GLenum::GL_RGB9_E5, GLenum::GL_SRGB8, GLenum::GL_RGB8UI, GLenum::GL_COMPRESSED_RGB})}; 
    auto texFormat {(texFormats)[(texFormatIdx)]};     
            av::init();
        auto ctx {av::FormatContext()}; 
        auto fn {positional.at(0)}; 
    spdlog::info("open video file  fn='{}'", fn);
    ctx.openInput(fn); 
    ctx.findStreamInfo();
    spdlog::info("stream info  ctx.seekable()='{}'  ctx.startTime().seconds()='{}'  ctx.duration().seconds()='{}'  ctx.streamsCount()='{}'", ctx.seekable(), ctx.startTime().seconds(), ctx.duration().seconds(), ctx.streamsCount());
    ctx.seek({static_cast<long int>(floor((100)*((0.50F)*(ctx.duration().seconds())))), {1, 100}});
        ssize_t videoStream = -1; 
    av::Stream vst; 
    std::error_code ec; 
    for ( decltype((0)+(ctx.streamsCount())+(1)) i = 0;(i)<(ctx.streamsCount());(i)+=(1) ) {
                        auto st {ctx.stream(i)}; 
        if ( (AVMEDIA_TYPE_VIDEO)==(st.mediaType()) ) {
                                                (videoStream)=(i);
            (vst)=(st); 
            break; 
}  
} 
    if ( vst.isNull() ) {
                        spdlog::info("Video stream not found"); 
} 
        av::VideoDecoderContext vdec; 
    if ( vst.isValid() ) {
                                (vdec)=(av::VideoDecoderContext(vst)); 
                auto codec {av::findDecodingCodec((vdec.raw())->(codec_id))}; 
        vdec.setCodec(codec);
        vdec.setRefCountedFrames(true);
        vdec.open({{"threads", "1"}}, av::Codec(), ec);
        if ( ec ) {
                                    spdlog::info("can't open codec"); 
}   
}     
            auto *window {([&] ()-> GLFWwindow* {
                spdlog::info("initialize GLFW3");
                if ( !(glfwInit()) ) {
                                    spdlog::info("glfwInit failed"); 
} 
                glfwWindowHint(GLFW_VISIBLE, true);
                glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
                glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
                glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, true);
                glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
                spdlog::info("create GLFW3 window");
                        const auto startWidth {800}; 
        const auto startHeight {600}; 
        auto window {glfwCreateWindow(startWidth, startHeight, "glfw", nullptr, nullptr)}; 
        if ( !(window) ) {
                                    spdlog::info("can't create glfw window"); 
} 
        spdlog::info("initialize GLFW3 context for window");
        glfwMakeContextCurrent(window);
                // configure Vsync, 1 locks to 60Hz, FIXME: i should really check glfw errors 
        glfwSwapInterval(0); 
        return window; 
})()};  
            auto width {int(0)}; 
    auto height {int(0)}; 
        spdlog::info("initialize glbinding");
    // if second arg is false: lazy function pointer loading 
    glbinding::initialize(glfwGetProcAddress, false);
    {
                        const float r {0.40F};  
                        const float g {0.40F};  
                        const float b {0.20F};  
                        const float a {1.0F};  
                glClearColor(r, g, b, a);
}  
        spdlog::info("initialize ImGui");
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
        auto io {ImGui::GetIO()}; 
        (io.ConfigFlags)=((io.ConfigFlags)||(ImGuiConfigFlags_NavEnableKeyboard));  
    ImGui::StyleColorsLight();
    {
                        const auto installCallbacks {true}; 
        ImGui_ImplGlfw_InitForOpenGL(window, installCallbacks); 
} 
        const auto glslVersion {"#version 150"}; 
    ImGui_ImplOpenGL3_Init(glslVersion);  
        const auto radius {10.F}; 
        bool video_is_initialized_p = false; 
    int image_width = 0; 
    int image_height = 0; 
    GLuint image_texture = 0;  
    spdlog::info("start loop");
    while ( !(glfwWindowShouldClose(window)) ) {
                glfwPollEvents();
                        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
                auto showDemoWindow {true}; 
        ImGui::ShowDemoWindow(&showDemoWindow);  
                ([&width,&height,window] (){
                        // react to changing window size 
                                    auto oldwidth {width}; 
            auto oldheight {height}; 
            glfwGetWindowSize(window, &width, &height);
            if ( ((width)!=(oldwidth)) | ((height)!=(oldheight)) ) {
                                                spdlog::info("window size has changed  width='{}'  height='{}'", width, height);
                glViewport(0, 0, width, height); 
}  
})();
                {
                        std::error_code ec; 
                        av::Packet pkt; 
                        while ( (pkt)=(ctx.readPacket(ec)) ) {
                                if ( ec ) {
                                                            spdlog::info("packet reading error  ec.message()='{}'", ec.message()); 
} 
                                if ( !((videoStream)==(pkt.streamIndex())) ) {
                                                            continue; 
} 
                                                auto ts {pkt.ts()}; 
                                auto frame {vdec.decode(pkt, ec)}; 
                if ( ec ) {
                                                            spdlog::info("error  ec.message()='{}'", ec.message()); 
} 
                                (ts)=(frame.pts()); 
                if ( (frame.isComplete()) & (frame.isValid()) ) {
                                                                                auto *data {frame.data(0)}; 
                                        (image_width)=(frame.(raw())->((linesize)[(0)]));
                    (image_height)=(frame.height()); 
                                        auto init_width {image_width}; 
                    auto init_height {image_height}; 
                    if ( !video_is_initialized_p ) {
                                                                        // initialize texture for video frames 
                        glGenTextures(1, &image_texture);
                        glBindTexture(GL_TEXTURE_2D, image_texture);
                        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
                        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
                        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                                                spdlog::info("prepare texture  init_width='{}'  image_width='{}'  image_height='{}'  frame.width()='{}'", init_width, image_width, image_height, frame.width());
                        glTexImage2D(GL_TEXTURE_2D, 0, texFormat, image_width, image_height, 0, GLenum::GL_LUMINANCE, GL_UNSIGNED_BYTE, nullptr);
                        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, image_width, image_height, GLenum::GL_LUMINANCE, GL_UNSIGNED_BYTE, data); 
                                                (video_is_initialized_p)=(true);  
} else {
                                                                        // update texture with new frame 
                        glBindTexture(GL_TEXTURE_2D, image_texture);
                        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, image_width, image_height, GLenum::GL_LUMINANCE, GL_UNSIGNED_BYTE, data); 
}  
                    break;  
}   
} 
                                    // draw frame 
                        ImGui::Begin("video texture");
            ImGui::Text("width = %d", image_width);
            ImGui::Text("fn = %s", fn.c_str());
            ImGui::Image(reinterpret_cast<void*>(static_cast<intptr_t>(image_texture)), ImVec2(static_cast<float>(image_width), static_cast<float>(image_height)));
                        auto val_old {static_cast<float>(pkt.ts().seconds())}; 
            auto val {val_old}; 
            ImGui::SliderFloat("time", &val, static_cast<float>(ctx.startTime().seconds()), static_cast<float>(ctx.duration().seconds()), "%.3f");
            if ( !((val)==(val_old)) ) {
                                                // perform seek operation 
                ctx.seek({static_cast<long int>(floor((1000)*(val))), {1, 1000}}); 
}  
            ImGui::End(); 
            ImGui::Render();
            glClear(GL_COLOR_BUFFER_BIT);
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
            glfwSwapBuffers(window); 
} 
}   
            spdlog::info("Shutdown ImGui and GLFW3");
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate(); 
        return 0;
}
 