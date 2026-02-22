// ============================================================================
// Minimal Modular Modern C++ Dense Point Cloud Viewer with Eye-Dome Lighting
// ============================================================================
// Dependencies: GLFW, GLAD, GLM, Dear ImGui
// C++ Standard: C++17 or higher
// Paradigm: Almost Always Auto (AAA)
// ============================================================================

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>

// ----------------------------------------------------------------------------
// Shader Source Code (Embedded as Raw String Literals for single-file design)
// ----------------------------------------------------------------------------

constexpr auto geometry_vertex_shader_src = R"glsl(
#version 330 core
layout(location = 0) in vec3 aPos;
uniform mat4 uMVP;
void main() {
    gl_Position = uMVP * vec4(aPos, 1.0);
}
)glsl";

constexpr auto geometry_fragment_shader_src = R"glsl(
#version 330 core
out vec4 FragColor;
void main() {
    // Neutral base color for unlit point cloud (No color attributes exist)
    FragColor = vec4(0.85, 0.85, 0.85, 1.0); 
}
)glsl";

constexpr auto edl_vertex_shader_src = R"glsl(
#version 330 core
out vec2 vTexCoords;
void main() {
    // Generate full-screen quad directly from gl_VertexID 
    // This entirely avoids the need to bind a dedicated VBO for post-processing.
    float x = -1.0 + float((gl_VertexID & 1) << 2);
    float y = -1.0 + float((gl_VertexID & 2) << 1);
    vTexCoords.x = (x + 1.0) * 0.5;
    vTexCoords.y = (y + 1.0) * 0.5;
    gl_Position = vec4(x, y, 0.0, 1.0);
}
)glsl";

constexpr auto edl_fragment_shader_src = R"glsl(
#version 330 core
out vec4 FragColor;
in vec2 vTexCoords;

uniform sampler2D uColorTexture;
uniform sampler2D uDepthTexture;

uniform float uScreenWidth;
uniform float uScreenHeight;
uniform float uEdlStrength;
uniform float uEdlRadius;

void main() {
    vec4 baseColor = texture(uColorTexture, vTexCoords);
    float depth = texture(uDepthTexture, vTexCoords).r;

    // Background threshold rejection (do not apply EDL to the empty skybox)
    if (depth > 0.99999) {
        FragColor = baseColor;
        return;
    }

    vec2 texelSize = 1.0 / vec2(uScreenWidth, uScreenHeight);
    vec2 radiusSize = texelSize * uEdlRadius;

    // Standard cross-shaped neighborhood search in pixel space
    vec2 offsets = vec2(
        vec2( 0.0,  1.0), // top
        vec2( 0.0, -1.0), // bottom
        vec2( 1.0,  0.0), // right
        vec2(-1.0,  0.0)  // left
    );

    float sum = 0.0;
    
    // Evaluate depth discontinuities (Non-linear depth shortcut)
    for(int i = 0; i < 4; i++) {
        vec2 neighborCoords = vTexCoords + offsets[i] * radiusSize;
        float neighborDepth = texture(uDepthTexture, neighborCoords).r;
        
        // Background threshold for neighbors
        if (neighborDepth > 0.99999) { neighborDepth = 0.0; }
        
        if(neighborDepth!= 0.0) {
            // max(0, depth - neighborDepth):
            // In OpenGL, closer objects have numerically smaller depth values.
            // If current depth > neighborDepth, current pixel is physically BEHIND neighbor.
            // This logic creates a shadow on the surface directly behind an edge.
            sum += max(0.0, depth - neighborDepth);
        }
    }

    // Average the obscurance sum across the 4 neighbors
    float res = sum / 4.0;
    
    // Exponential decay to determine final shading factor (C = 300.0)
    float shade = exp(-res * 300.0 * uEdlStrength);

    FragColor = vec4(baseColor.rgb * shade, baseColor.a);
}
)glsl";

// ----------------------------------------------------------------------------
// Utility Functions
// ----------------------------------------------------------------------------

auto compile_shader(GLuint type, const char* source) -> GLuint {
    auto shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    auto success = GLint{};
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        auto infoLog = std::string(512, '\0');
        glGetShaderInfoLog(shader, 512, nullptr, infoLog.data());
        std::cerr << "Shader Compilation Error:\n" << infoLog << "\n";
    }
    return shader;
}

auto create_shader_program(const char* v_src, const char* f_src) -> GLuint {
    auto vertex_shader = compile_shader(GL_VERTEX_SHADER, v_src);
    auto fragment_shader = compile_shader(GL_FRAGMENT_SHADER, f_src);

    auto program = glCreateProgram();
    glAttachShader(program, vertex_shader);
    glAttachShader(program, fragment_shader);
    glLinkProgram(program);

    glDeleteShader(vertex_shader);
    glDeleteShader(fragment_shader);
    return program;
}

auto load_xyz_point_cloud(const std::string& filepath) -> std::vector<float> {
    auto points = std::vector<float>{};
    auto file = std::ifstream{filepath};
    
    if (!file.is_open()) {
        std::cerr << "Failed to open XYZ file: " << filepath << "\n";
        return points;
    }

    auto line = std::string{};
    // Read sequentially via streams. std::vector dynamic resizing handles memory allocations safely.
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        auto ss = std::istringstream{line};
        auto x = float{}, y = float{}, z = float{};
        if (ss >> x >> y >> z) {
            points.push_back(x);
            points.push_back(y);
            points.push_back(z);
        }
    }
    return points;
}

// ----------------------------------------------------------------------------
// Main Application Execution Loop
// ----------------------------------------------------------------------------

auto main() -> int {
    // 1. GLFW and OpenGL Context Initialization
    if (!glfwInit()) return -1;

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    auto window_width = int{1280};
    auto window_height = int{720};
    auto window = glfwCreateWindow(window_width, window_height, "Modern C++ AAA EDL Viewer", nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    
    if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress))) {
        std::cerr << "Failed to initialize GLAD\n";
        return -1;
    }

    glEnable(GL_DEPTH_TEST);
    // Explicitly set depth function for strict Z-testing
    glDepthFunc(GL_LESS);

    // 2. Load Data and Configure Geometric State via VAO/VBO
    auto point_data = load_xyz_point_cloud("scan.txt"); // Assuming target point cloud is named scan.txt
    if(point_data.empty()) {
        // Fallback fake data array if no target file is provided to prevent application crash
        point_data = {
            0.0f, 0.0f, 0.0f,
            0.1f, 0.1f, -0.1f,
           -0.1f, 0.2f,  0.1f
        };
    }

    auto vao = GLuint{};
    auto vbo = GLuint{};
    
    // Utilizing AAA paradigm variables ensures memory safety prior to GL calls
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    
    // Upload entire contiguous vector in one massive, perfectly optimized memory chunk
    glBufferData(GL_ARRAY_BUFFER, point_data.size() * sizeof(float), point_data.data(), GL_STATIC_DRAW);
    
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), reinterpret_cast<void*>(0));
    glEnableVertexAttribArray(0);

    // 3. Configure Off-Screen Framebuffer Object (FBO) for Deferred Post-Processing
    auto fbo = GLuint{};
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);

    auto color_texture = GLuint{};
    glGenTextures(1, &color_texture);
    glBindTexture(GL_TEXTURE_2D, color_texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, window_width, window_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color_texture, 0);

    auto depth_texture = GLuint{};
    glGenTextures(1, &depth_texture);
    glBindTexture(GL_TEXTURE_2D, depth_texture);
    
    // Crucial Architecture Decision: 32-bit float for high precision depth differencing in EDL math
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, window_width, window_height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depth_texture, 0);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER)!= GL_FRAMEBUFFER_COMPLETE) {
        std::cerr << "Framebuffer Architecture Error: FBO is not complete!\n";
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // 4. Compile Monolithic Shader Programs
    auto geom_program = create_shader_program(geometry_vertex_shader_src, geometry_fragment_shader_src);
    auto edl_program = create_shader_program(edl_vertex_shader_src, edl_fragment_shader_src);

    // 5. Initialize Dear ImGui (Immediate Mode GUI avoiding gRPC overhead)
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    auto& io = ImGui::GetIO(); (void)io;
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330 core");

    // EDL Initial Configuration Parameters
    auto edl_radius = float{1.5f};
    auto edl_strength = float{2.0f};
    
    // Viewport and Orbital Camera Setup
    auto angle = float{0.0f};

    // 6. Main Render Loop
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        // Handle dynamic window resizing for the FBO texture memory allocation
        auto current_w = int{}, current_h = int{};
        glfwGetFramebufferSize(window, &current_w, &current_h);
        if (current_w!= window_width |

| current_h!= window_height) {
            window_width = current_w;
            window_height = current_h;
            
            glBindTexture(GL_TEXTURE_2D, color_texture);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, window_width, window_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
            glBindTexture(GL_TEXTURE_2D, depth_texture);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, window_width, window_height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
        }

        // --- ImGui New Frame Construction ---
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::Begin("Eye-Dome Lighting Settings");
        ImGui::SliderFloat("Radius", &edl_radius, 0.1f, 5.0f);
        ImGui::SliderFloat("Strength", &edl_strength, 0.1f, 10.0f);
        ImGui::Text("Points Rendered: %zu", point_data.size() / 3);
        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
        ImGui::End();

        // --- Pass 1: Render Raw Point Geometry to Deferred FBO ---
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glViewport(0, 0, window_width, window_height);
        
        // Clear background to distant black (Hardware depth automatically resolves to 1.0)
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f); 
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glUseProgram(geom_program);

        // Orbital camera projection matrix math execution
        angle += 0.01f;
        auto view = glm::lookAt(glm::vec3(sin(angle) * 5.0f, 2.0f, cos(angle) * 5.0f), 
                                glm::vec3(0.0f, 0.0f, 0.0f), 
                                glm::vec3(0.0f, 1.0f, 0.0f));
        auto proj = glm::perspective(glm::radians(45.0f), (float)window_width / (float)window_height, 0.1f, 100.0f);
        auto mvp = proj * view;

        auto mvp_loc = glGetUniformLocation(geom_program, "uMVP");
        glUniformMatrix4fv(mvp_loc, 1, GL_FALSE, glm::value_ptr(mvp));

        glBindVertexArray(vao);
        // Assuming every screen pixel is covered by density, GL_POINTS is algorithmically sufficient
        glDrawArrays(GL_POINTS, 0, point_data.size() / 3);

        // --- Pass 2: EDL Screen-Space Post-Processing to Default System Framebuffer ---
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0, 0, window_width, window_height);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glUseProgram(edl_program);

        // Bind Deferred Textures to Shader Uniforms
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, color_texture);
        glUniform1i(glGetUniformLocation(edl_program, "uColorTexture"), 0);

        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, depth_texture);
        glUniform1i(glGetUniformLocation(edl_program, "uDepthTexture"), 1);

        glUniform1f(glGetUniformLocation(edl_program, "uScreenWidth"), static_cast<float>(window_width));
        glUniform1f(glGetUniformLocation(edl_program, "uScreenHeight"), static_cast<float>(window_height));
        glUniform1f(glGetUniformLocation(edl_program, "uEdlRadius"), edl_radius);
        glUniform1f(glGetUniformLocation(edl_program, "uEdlStrength"), edl_strength);

        // Draw full-screen quad via non-physical vertex ID shortcut (3 vertices spanning screen, clipping trims excess)
        glDrawArrays(GL_TRIANGLES, 0, 3);

        // --- ImGui Final Render Pass Overlay ---
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    // 7. Context Teardown and Memory Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo);
    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &color_texture);
    glDeleteTextures(1, &depth_texture);
    glDeleteProgram(geom_program);
    glDeleteProgram(edl_program);

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
