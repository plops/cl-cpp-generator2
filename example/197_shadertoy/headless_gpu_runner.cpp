#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <chrono>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>

#define GL_GLEXT_PROTOTYPES
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GL/gl.h>
#include <GL/glext.h>
#include <png.h>

// Save RGBA buffer to PNG file
bool save_png(const char *filename, int width, int height, const unsigned char *buffer) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "Error: Could not open %s for writing\n", filename);
        return false;
    }

    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr) {
        fclose(fp);
        return false;
    }

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_write_struct(&png_ptr, NULL);
        fclose(fp);
        return false;
    }

    if (setjmp(png_jmpbuf(png_ptr))) {
        png_destroy_write_struct(&png_ptr, &info_ptr);
        fclose(fp);
        return false;
    }

    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height,
                 8, PNG_COLOR_TYPE_RGBA, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

    png_write_info(png_ptr, info_ptr);

    // PNG coordinates start from top-left, OpenGL FBO is bottom-left (so we flip rows)
    std::vector<png_bytep> row_pointers(height);
    for (int y = 0; y < height; y++) {
        row_pointers[y] = (png_bytep)&buffer[(height - 1 - y) * width * 4];
    }

    png_write_image(png_ptr, row_pointers.data());
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
    return true;
}

std::string read_file(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        fprintf(stderr, "Warning: Could not open file: %s\n", path.c_str());
        return "";
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

GLuint compile_shader(GLenum type, const std::string& source, const std::string& debug_name) {
    GLuint shader = glCreateShader(type);
    const char* src_cstr = source.c_str();
    glShaderSource(shader, 1, &src_cstr, NULL);
    glCompileShader(shader);

    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[2048];
        glGetShaderInfoLog(shader, sizeof(infoLog), NULL, infoLog);
        fprintf(stderr, "Shader compilation error in [%s]:\n%s\n", debug_name.c_str(), infoLog);
        glDeleteShader(shader);
        return 0;
    }
    return shader;
}

GLuint create_program(const std::string& vert_src, const std::string& frag_src, const std::string& name) {
    GLuint vert = compile_shader(GL_VERTEX_SHADER, vert_src, name + " (vert)");
    if (!vert) return 0;
    GLuint frag = compile_shader(GL_FRAGMENT_SHADER, frag_src, name + " (frag)");
    if (!frag) { glDeleteShader(vert); return 0; }

    GLuint program = glCreateProgram();
    glAttachShader(program, vert);
    glAttachShader(program, frag);
    glLinkProgram(program);

    GLint success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[2048];
        glGetProgramInfoLog(program, sizeof(infoLog), NULL, infoLog);
        fprintf(stderr, "Program linking error in [%s]:\n%s\n", name.c_str(), infoLog);
        glDeleteShader(vert);
        glDeleteShader(frag);
        glDeleteProgram(program);
        return 0;
    }

    glDeleteShader(vert);
    glDeleteShader(frag);
    return program;
}

const char* VERTEX_SHADER_SRC = R"(#version 330 core
layout (location = 0) in vec2 aPos;
out vec2 v_uv;
void main() {
    v_uv = (aPos + 1.0) * 0.5;
    gl_Position = vec4(aPos, 0.0, 1.0);
}
)";

std::string build_shadertoy_frag(const std::string& common_src, const std::string& user_glsl, bool is_buf0) {
    std::stringstream ss;
    ss << "#version 330 core\n";
    ss << "#extension GL_ARB_explicit_attrib_location : enable\n";
    ss << "out vec4 FragColor;\n";
    ss << "in vec2 v_uv;\n";
    ss << "uniform vec3 iResolution;\n";
    ss << "uniform float iTime;\n";
    ss << "uniform float iTimeDelta;\n";
    ss << "uniform int iFrame;\n";
    ss << "uniform vec4 iMouse;\n";
    ss << "uniform vec4 iDate;\n";
    ss << "uniform sampler2D iChannel0;\n";
    ss << "uniform sampler2D iChannel1;\n";
    ss << "uniform sampler2D iChannel2;\n";
    ss << "uniform sampler2D iChannel3;\n";
    ss << "uniform sampler2D iKeyboard;\n";
    ss << "\n// --- COMMON ---\n";
    ss << common_src << "\n";
    ss << "\n// --- SHADER BODY ---\n";
    ss << user_glsl << "\n";
    ss << "\nvoid main() {\n";
    ss << "    vec4 col = vec4(0.0);\n";
    ss << "    vec2 fragCoord = gl_FragCoord.xy;\n";
    ss << "    mainImage(col, fragCoord);\n";
    ss << "    FragColor = col;\n";
    ss << "}\n";
    return ss.str();
}

struct FBO {
    GLuint fbo = 0;
    GLuint texture = 0;
    int width = 0;
    int height = 0;

    void init(int w, int h, GLint internal_format = GL_RGBA32F) {
        width = w;
        height = h;
        glGenFramebuffers(1, &fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);

        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexImage2D(GL_TEXTURE_2D, 0, internal_format, w, h, 0, GL_RGBA, GL_FLOAT, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0);

        GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
        if (status != GL_FRAMEBUFFER_COMPLETE) {
            fprintf(stderr, "FBO init failed: 0x%x\n", status);
        }
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    void destroy() {
        if (texture) glDeleteTextures(1, &texture);
        if (fbo) glDeleteFramebuffers(1, &fbo);
    }
};

int main(int argc, char** argv) {
    int res_w = 1280;
    int res_h = 720;
    int num_frames = 60;
    bool do_benchmark = false;
    int benchmark_frames = 500;
    std::string screenshot_file = "gpu_screenshot.png";
    std::string shader_dir = "vulkan-shadertoy-x11/launcher/shaders/shadertoy";

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--res" && i + 2 < argc) {
            res_w = atoi(argv[++i]);
            res_h = atoi(argv[++i]);
        } else if (arg == "--frames" && i + 1 < argc) {
            num_frames = atoi(argv[++i]);
        } else if (arg == "--screenshot" && i + 1 < argc) {
            screenshot_file = argv[++i];
        } else if (arg == "--benchmark") {
            do_benchmark = true;
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                benchmark_frames = atoi(argv[++i]);
            }
        } else if (arg == "--shader-dir" && i + 1 < argc) {
            shader_dir = argv[++i];
        } else if (arg == "--help") {
            printf("Usage: %s [options]\n", argv[0]);
            printf("  --res <W> <H>         Set render resolution (default 1280 720)\n");
            printf("  --frames <N>          Number of simulation frames (default 60)\n");
            printf("  --screenshot <file>   Save PNG screenshot (default gpu_screenshot.png)\n");
            printf("  --benchmark [N]       Run benchmark for N frames (default 500)\n");
            printf("  --shader-dir <path>   Directory with buf0.glsl, main_image.glsl\n");
            return 0;
        }
    }

    // 1. Initialize EGL with NVIDIA GPU
    EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (display == EGL_NO_DISPLAY) {
        fprintf(stderr, "Failed to get EGL display\n");
        return 1;
    }

    EGLint major, minor;
    if (!eglInitialize(display, &major, &minor)) {
        fprintf(stderr, "Failed to initialize EGL\n");
        return 1;
    }

    eglBindAPI(EGL_OPENGL_API);
    const EGLint configAttribs[] = {
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
        EGL_RED_SIZE, 8,
        EGL_GREEN_SIZE, 8,
        EGL_BLUE_SIZE, 8,
        EGL_ALPHA_SIZE, 8,
        EGL_NONE
    };

    EGLConfig config;
    EGLint numConfigs;
    eglChooseConfig(display, configAttribs, &config, 1, &numConfigs);

    const EGLint contextAttribs[] = {
        EGL_CONTEXT_MAJOR_VERSION, 4,
        EGL_CONTEXT_MINOR_VERSION, 6,
        EGL_NONE
    };

    EGLContext context = eglCreateContext(display, config, EGL_NO_CONTEXT, contextAttribs);
    if (context == EGL_NO_CONTEXT) {
        fprintf(stderr, "Failed to create OpenGL 4.6 context\n");
        return 1;
    }

    const EGLint pbufferAttribs[] = {
        EGL_WIDTH, 64,
        EGL_HEIGHT, 64,
        EGL_NONE
    };
    EGLSurface surface = eglCreatePbufferSurface(display, config, pbufferAttribs);
    eglMakeCurrent(display, surface, surface, context);

    printf("============================================================\n");
    printf("🚀 NVIDIA GPU HARDWARE ACCELERATION INITIALIZED\n");
    printf("============================================================\n");
    printf("  GPU Renderer    : %s\n", glGetString(GL_RENDERER));
    printf("  OpenGL Driver   : %s\n", glGetString(GL_VERSION));
    printf("  GLSL Version    : %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));
    printf("  Target Viewport : %dx%d (%.2f MP)\n", res_w, res_h, (res_w * res_h) / 1000000.0);
    printf("============================================================\n");

    // 2. Load Shaders
    std::string common_src = read_file(shader_dir + "/common.glsl");
    std::string buf0_src = read_file(shader_dir + "/buf0.glsl");
    std::string main_src = read_file(shader_dir + "/main_image.glsl");

    if (main_src.empty()) {
        fprintf(stderr, "Error: main_image.glsl not found in %s\n", shader_dir.c_str());
        return 1;
    }

    std::string full_buf0 = build_shadertoy_frag(common_src, buf0_src, true);
    std::string full_main = build_shadertoy_frag(common_src, main_src, false);

    GLuint prog_buf0 = 0;
    if (!buf0_src.empty()) {
        prog_buf0 = create_program(VERTEX_SHADER_SRC, full_buf0, "Buffer A (buf0)");
        if (!prog_buf0) {
            fprintf(stderr, "Failed to compile Buffer A shader\n");
            return 1;
        }
    }

    GLuint prog_main = create_program(VERTEX_SHADER_SRC, full_main, "Main Image (main_image)");
    if (!prog_main) {
        fprintf(stderr, "Failed to compile Main Image shader\n");
        return 1;
    }

    // 3. Set up Full-Screen Quad Geometry
    float quad_verts[] = {
        -1.0f, -1.0f,
         1.0f, -1.0f,
        -1.0f,  1.0f,
         1.0f,  1.0f,
    };
    GLuint vao, vbo;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad_verts), quad_verts, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glBindVertexArray(0);

    // Dummy keyboard texture
    GLuint tex_keyboard;
    glGenTextures(1, &tex_keyboard);
    glBindTexture(GL_TEXTURE_2D, tex_keyboard);
    std::vector<float> kb_data(256 * 3 * 4, 0.0f);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 256, 3, 0, GL_RGBA, GL_FLOAT, kb_data.data());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    // 4. Initialize Framebuffers (Ping-pong for Buffer A, plus Main Out)
    FBO fbo_buf0[2];
    fbo_buf0[0].init(res_w, res_h);
    fbo_buf0[1].init(res_w, res_h);

    FBO fbo_main;
    fbo_main.init(res_w, res_h, GL_RGBA8);

    auto render_frame = [&](int frame_idx, float current_time, float dt) {
        int read_buf = frame_idx % 2;
        int write_buf = (frame_idx + 1) % 2;

        // --- Pass 1: Render Buffer A ---
        if (prog_buf0) {
            glBindFramebuffer(GL_FRAMEBUFFER, fbo_buf0[write_buf].fbo);
            glViewport(0, 0, res_w, res_h);
            glUseProgram(prog_buf0);

            glUniform3f(glGetUniformLocation(prog_buf0, "iResolution"), (float)res_w, (float)res_h, 1.0f);
            glUniform1f(glGetUniformLocation(prog_buf0, "iTime"), current_time);
            glUniform1f(glGetUniformLocation(prog_buf0, "iTimeDelta"), dt);
            glUniform1i(glGetUniformLocation(prog_buf0, "iFrame"), frame_idx);
            glUniform4f(glGetUniformLocation(prog_buf0, "iMouse"), 0.0f, 0.0f, 0.0f, 0.0f);
            glUniform4f(glGetUniformLocation(prog_buf0, "iDate"), 2026.0f, 8.0f, 30.0f, current_time);

            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, fbo_buf0[read_buf].texture);
            glUniform1i(glGetUniformLocation(prog_buf0, "iChannel0"), 0);

            glActiveTexture(GL_TEXTURE4);
            glBindTexture(GL_TEXTURE_2D, tex_keyboard);
            glUniform1i(glGetUniformLocation(prog_buf0, "iKeyboard"), 4);

            glBindVertexArray(vao);
            glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        }

        // --- Pass 2: Render Main Image ---
        glBindFramebuffer(GL_FRAMEBUFFER, fbo_main.fbo);
        glViewport(0, 0, res_w, res_h);
        glUseProgram(prog_main);

        glUniform3f(glGetUniformLocation(prog_main, "iResolution"), (float)res_w, (float)res_h, 1.0f);
        glUniform1f(glGetUniformLocation(prog_main, "iTime"), current_time);
        glUniform1f(glGetUniformLocation(prog_main, "iTimeDelta"), dt);
        glUniform1i(glGetUniformLocation(prog_main, "iFrame"), frame_idx);
        glUniform4f(glGetUniformLocation(prog_main, "iMouse"), 0.0f, 0.0f, 0.0f, 0.0f);
        glUniform4f(glGetUniformLocation(prog_main, "iDate"), 2026.0f, 8.0f, 30.0f, current_time);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, prog_buf0 ? fbo_buf0[write_buf].texture : 0);
        glUniform1i(glGetUniformLocation(prog_main, "iChannel0"), 0);

        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    };

    printf("\n🎬 Running simulation (%d frames at 60 FPS)...\n", num_frames);
    float dt = 1.0f / 60.0f;
    for (int f = 0; f < num_frames; f++) {
        render_frame(f, f * dt, dt);
    }
    glFinish();

    // 5. Read back pixels and save screenshot
    std::vector<unsigned char> pixel_buffer(res_w * res_h * 4);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_main.fbo);
    glReadPixels(0, 0, res_w, res_h, GL_RGBA, GL_UNSIGNED_BYTE, pixel_buffer.data());

    if (save_png(screenshot_file.c_str(), res_w, res_h, pixel_buffer.data())) {
        printf("📸 Saved high-resolution GPU screenshot to: %s\n", screenshot_file.c_str());
    }

    // 6. Benchmarking Mode
    if (do_benchmark) {
        printf("\n============================================================\n");
        printf("⚡ GPU BENCHMARK & PERFORMANCE MEASUREMENTS\n");
        printf("============================================================\n");
        printf("  Warmup frames       : 50\n");
        printf("  Measurement frames  : %d\n", benchmark_frames);
        printf("  Resolution          : %dx%d (%.2f MPixels/frame)\n", res_w, res_h, (res_w * res_h) / 1000000.0);

        // Warmup
        for (int f = 0; f < 50; f++) {
            render_frame(f, f * dt, dt);
        }
        glFinish();

        std::vector<double> frame_times_ms(benchmark_frames);
        auto t_start_total = std::chrono::high_resolution_clock::now();

        for (int f = 0; f < benchmark_frames; f++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            render_frame(f, f * dt, dt);
            glFinish(); // GPU sync for accurate per-frame timing
            auto t1 = std::chrono::high_resolution_clock::now();
            frame_times_ms[f] = std::chrono::duration<double, std::milli>(t1 - t0).count();
        }

        auto t_end_total = std::chrono::high_resolution_clock::now();
        double total_time_sec = std::chrono::duration<double>(t_end_total - t_start_total).count();

        // Statistical Analysis
        std::sort(frame_times_ms.begin(), frame_times_ms.end());
        double sum_ms = std::accumulate(frame_times_ms.begin(), frame_times_ms.end(), 0.0);
        double avg_ms = sum_ms / benchmark_frames;
        double min_ms = frame_times_ms.front();
        double max_ms = frame_times_ms.back();
        double p50_ms = frame_times_ms[benchmark_frames * 50 / 100];
        double p95_ms = frame_times_ms[benchmark_frames * 95 / 100];
        double p99_ms = frame_times_ms[benchmark_frames * 99 / 100];
        double fps = benchmark_frames / total_time_sec;
        double mpixels_per_sec = (fps * res_w * res_h) / 1000000.0;

        printf("------------------------------------------------------------\n");
        printf("  Average Throughput  : %8.2f FPS\n", fps);
        printf("  Pixel Rate          : %8.2f MegaPixels / sec\n", mpixels_per_sec);
        printf("  Mean Frame Time     : %8.3f ms\n", avg_ms);
        printf("  Median Frame Time   : %8.3f ms\n", p50_ms);
        printf("  Min Frame Time      : %8.3f ms\n", min_ms);
        printf("  Max Frame Time      : %8.3f ms\n", max_ms);
        printf("  95th Percentile     : %8.3f ms\n", p95_ms);
        printf("  99th Percentile     : %8.3f ms\n", p99_ms);
        printf("============================================================\n");
    }

    // Cleanup
    fbo_buf0[0].destroy();
    fbo_buf0[1].destroy();
    fbo_main.destroy();
    glDeleteTextures(1, &tex_keyboard);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    if (prog_buf0) glDeleteProgram(prog_buf0);
    glDeleteProgram(prog_main);

    eglDestroySurface(display, surface);
    eglDestroyContext(display, context);
    eglTerminate(display);

    return 0;
}
