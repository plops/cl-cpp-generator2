#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <limits>
#include <algorithm>

// ============================================================================
// SHADER SOURCE CODE DEFINITIONS
// ============================================================================

// 1. Geometry Pass Vertex Shader
// Projects the raw 3D points into screen space.
const auto geometryVS = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
uniform mat4 mvp;

void main() {
    gl_Position = mvp * vec4(aPos, 1.0);
}
)";

// 2. Geometry Pass Fragment Shader
// Outputs a flat base color. The vital depth data is written automatically
// by the OpenGL pipeline into the attached depth texture.
const auto geometryFS = R"(
#version 330 core
layout (location = 0) out vec4 FragColor;
uniform vec3 baseColor;

void main() {
    FragColor = vec4(baseColor, 1.0);
}
)";

// 3. Post-Processing Vertex Shader (Fullscreen Quad)
// Renders a flat quad covering the entire screen to host the post-processing.
const auto edlVS = R"(
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aTexCoords;

out vec2 TexCoords;

void main() {
    TexCoords = aTexCoords;
    gl_Position = vec4(aPos.x, aPos.y, 0.0, 1.0);
}
)";

// 4. Post-Processing Fragment Shader (Eye-Dome Lighting)
// Implements the depth-disparity non-photorealistic shading logic.
const auto edlFS = R"(
#version 330 core
out vec4 FragColor;
in vec2 TexCoords;

uniform sampler2D colorTexture;
uniform sampler2D depthTexture;

uniform float edlStrength;
uniform float edlRadius;
uniform float edlOffset;
uniform vec2 resolution;
uniform float zNear;
uniform float zFar;

// Converts hyperbolic hardware depth back into linear eye-space depth
float linearizeDepth(float depth) {
    float z = depth * 2.0 - 1.0;
    return (2.0 * zNear * zFar) / (zFar + zNear - z * (zFar - zNear));
}

void main() {
    float rawDepth = texture(depthTexture, TexCoords).r;

    // If the pixel is the clear color (depth == 1.0), render the background
    if (rawDepth >= 1.0) {
        FragColor = vec4(0.15, 0.15, 0.15, 1.0); // Dark gray background
        return;
    }

    vec4 baseColor = texture(colorTexture, TexCoords);
    float depth = linearizeDepth(rawDepth);

    vec2 texelSize = 1.0 / resolution;

    // Orthogonal cross-filter sampling to optimize texture fetches
    vec2 offsets[4] = vec2[](
        vec2(edlRadius, 0.0),
        vec2(-edlRadius, 0.0),
        vec2(0.0, edlRadius),
        vec2(0.0, -edlRadius)
    );

    float sum = 0.0;

    for(int i = 0; i < 4; i++) {
        vec2 sampleCoords = TexCoords + offsets[i] * texelSize;
        float neighborRawDepth = texture(depthTexture, sampleCoords).r;

        // Massive penalty for borders against the skybox to create strong outlines
        if (neighborRawDepth >= 1.0) {
            sum += 10.0;
        } else {
            float neighborDepth = linearizeDepth(neighborRawDepth);
            // Calculate positive depth disparities, applying the noise-reduction offset
            sum += max(0.0, depth - neighborDepth - edlOffset);
        }
    }

    // Average the occlusion and apply exponential decay
    float factor = sum / 4.0;
    float shade = exp(-factor * 300.0 * edlStrength);

    FragColor = vec4(baseColor.rgb * shade, baseColor.a);
}
)";

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

auto compileShader(GLenum type, const char* source) -> GLuint {
    auto shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    auto success = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        auto infoLog = std::string(512, '\0');
        glGetShaderInfoLog(shader, 512, nullptr, infoLog.data());
        std::cerr << "Shader Compilation Failed:\n" << infoLog << std::endl;
    }
    return shader;
}

auto createProgram(const char* vsSource, const char* fsSource) -> GLuint {
    auto vs = compileShader(GL_VERTEX_SHADER, vsSource);
    auto fs = compileShader(GL_FRAGMENT_SHADER, fsSource);

    auto program = glCreateProgram();
    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glLinkProgram(program);

    glDeleteShader(vs);
    glDeleteShader(fs);
    return program;
}

// Parses raw XYZ text data and translates the bounding volume to the origin
auto loadPointCloud(const std::string& filepath, glm::vec3& outCenter) -> std::vector<glm::vec3> {
    auto points = std::vector<glm::vec3>{};
    auto file = std::ifstream{filepath};

    if (!file.is_open()) {
        std::cerr << "Warning: Failed to open point cloud file: " << filepath << std::endl;
        return points;
    }

    auto line = std::string{};
    auto minBounds = glm::vec3(std::numeric_limits<float>::max());
    auto maxBounds = glm::vec3(std::numeric_limits<float>::lowest());

    while (std::getline(file, line)) {
        auto ss = std::istringstream{line};
        auto p = glm::vec3{};
        if (ss >> p.x >> p.y >> p.z) {
            points.push_back(p);
            minBounds = glm::min(minBounds, p);
            maxBounds = glm::max(maxBounds, p);
        }
    }

    outCenter = (minBounds + maxBounds) * 0.5f;
    return points;
}

// ============================================================================
// MAIN APPLICATION
// ============================================================================

int main() {
    // 1. Initialize GLFW Context
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    auto window = glfwCreateWindow(1280, 720, "Eye-Dome Lighting Point Cloud Viewer", nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable V-Sync

    // 2. Initialize GLAD Function Pointers
    if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress))) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    // 3. Load Data & Normalize Coordinates
    auto center = glm::vec3(0.0f);
    auto points = loadPointCloud("../data/bunny.xyz", center);

    if(points.empty()) {
        std::cerr << "Generating dummy geometry for demonstration purposes." << std::endl;
        for(auto i = 0; i < 10000; i++) {
            points.push_back(glm::vec3(
                (rand() % 100) / 10.0f - 5.0f,
                (rand() % 100) / 10.0f - 5.0f,
                (rand() % 100) / 10.0f - 5.0f
            ));
        }
    } else {
        // Translation to origin prevents Z-fighting in georeferenced data
        for(auto& p : points) {
            p -= center;
        }
        std::cout << "Successfully loaded " << points.size() << " points." << std::endl;
    }

    // 4. Setup Geometry VAO/VBO for the Point Cloud
    auto vao = GLuint{};
    auto vbo = GLuint{};
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, points.size() * sizeof(glm::vec3), points.data(), GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), nullptr);
    glEnableVertexAttribArray(0);

    // Setup Fullscreen Quad for Post-Processing Pass
    float quadVertices[] = {
        // positions   // texCoords
        -1.0f,  1.0f,  0.0f, 1.0f,
        -1.0f, -1.0f,  0.0f, 0.0f,
         1.0f, -1.0f,  1.0f, 0.0f,

        -1.0f,  1.0f,  0.0f, 1.0f,
         1.0f, -1.0f,  1.0f, 0.0f,
         1.0f,  1.0f,  1.0f, 1.0f
    };

    auto quadVAO = GLuint{};
    auto quadVBO = GLuint{};
    glGenVertexArrays(1, &quadVAO);
    glGenBuffers(1, &quadVBO);

    glBindVertexArray(quadVAO);
    glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), reinterpret_cast<void*>(0));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), reinterpret_cast<void*>(2 * sizeof(float)));

    // 5. Allocate the Framebuffer Object (FBO)
    auto fbo = GLuint{};
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);

    auto colorTex = GLuint{};
    glGenTextures(1, &colorTex);
    glBindTexture(GL_TEXTURE_2D, colorTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 1280, 720, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, colorTex, 0);

    auto depthTex = GLuint{};
    glGenTextures(1, &depthTex);
    glBindTexture(GL_TEXTURE_2D, depthTex);
    // Crucial: 32-bit float requirement for depth evaluation precision
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, 1280, 720, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthTex, 0);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER)!= GL_FRAMEBUFFER_COMPLETE) {
        std::cerr << "Critical Error: Framebuffer architecture incomplete!" << std::endl;
        return -1;
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // 6. Compile Pipeline Shaders
    auto geomProgram = createProgram(geometryVS, geometryFS);
    auto edlProgram = createProgram(edlVS, edlFS);

    // 7. Initialize Dear ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330 core");

    // Configurable Runtime Parameters
    auto edlStrength = 0.8f;
    auto edlRadius = 1.5f;
    auto edlOffset = 0.001f;
    auto pointSize = 3.0f;
    auto baseColor = glm::vec3(0.7f, 0.8f, 0.9f);

    auto cameraDist = 20.0f;
    auto rotationY = 0.0f;
    auto rotationX = 0.5f;

    auto zNear = 0.1f;
    auto zFar = 1000.0f;

    // 8. Primary Render Loop
    glEnable(GL_DEPTH_TEST);

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        auto width = 0;
        auto height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        if(width == 0 || height == 0) continue;

        glViewport(0, 0, width, height);

        // Dynamically reallocate FBO textures if the user resizes the window
        glBindTexture(GL_TEXTURE_2D, colorTex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
        glBindTexture(GL_TEXTURE_2D, depthTex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);

        // ====================================================================
        // PASS 1: Geometry Rendering
        // ====================================================================
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        // Clear color to pure white representing infinite depth space (1.0)
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glUseProgram(geomProgram);

        auto proj = glm::perspective(glm::radians(60.0f), static_cast<float>(width)/height, zNear, zFar);
        auto view = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, -cameraDist));
        view = glm::rotate(view, rotationX, glm::vec3(1.0f, 0.0f, 0.0f));
        view = glm::rotate(view, rotationY, glm::vec3(0.0f, 1.0f, 0.0f));
        auto mvp = proj * view;

        glUniformMatrix4fv(glGetUniformLocation(geomProgram, "mvp"), 1, GL_FALSE, glm::value_ptr(mvp));
        glUniform3fv(glGetUniformLocation(geomProgram, "baseColor"), 1, glm::value_ptr(baseColor));

        glPointSize(pointSize);
        glBindVertexArray(vao);
        glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(points.size()));

        // ====================================================================
        // PASS 2: Post-Processing (Eye-Dome Lighting)
        // ====================================================================
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glUseProgram(edlProgram);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, colorTex);
        glUniform1i(glGetUniformLocation(edlProgram, "colorTexture"), 0);

        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, depthTex);
        glUniform1i(glGetUniformLocation(edlProgram, "depthTexture"), 1);

        // Push ImGui tuned variables to the GPU
        glUniform1f(glGetUniformLocation(edlProgram, "edlStrength"), edlStrength);
        glUniform1f(glGetUniformLocation(edlProgram, "edlRadius"), edlRadius);
        glUniform1f(glGetUniformLocation(edlProgram, "edlOffset"), edlOffset);
        glUniform2f(glGetUniformLocation(edlProgram, "resolution"), static_cast<float>(width), static_cast<float>(height));
        glUniform1f(glGetUniformLocation(edlProgram, "zNear"), zNear);
        glUniform1f(glGetUniformLocation(edlProgram, "zFar"), zFar);

        glBindVertexArray(quadVAO);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        // ====================================================================
        // PASS 3: Immediate Mode UI Overlay
        // ====================================================================
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::Begin("Eye-Dome Configuration");
        ImGui::Text("Non-Photorealistic Shading Settings");
        ImGui::Separator();
        ImGui::SliderFloat("EDL Strength", &edlStrength, 0.0f, 3.0f, "%.2f");
        ImGui::SliderFloat("EDL Radius", &edlRadius, 0.0f, 10.0f, "%.1f px");
        ImGui::SliderFloat("EDL Offset", &edlOffset, 0.0f, 0.01f, "%.4f");

        ImGui::Spacing();
        ImGui::Text("Geometry Settings");
        ImGui::Separator();
        ImGui::SliderFloat("Point Size", &pointSize, 1.0f, 15.0f, "%.1f px");
        ImGui::ColorEdit3("Base Color", glm::value_ptr(baseColor));

        ImGui::Spacing();
        ImGui::Text("Camera Settings");
        ImGui::Separator();
        ImGui::SliderFloat("Zoom Distance", &cameraDist, 0.1f, 200.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("Yaw Rotation", &rotationY, 0.0f, 6.28f);
        ImGui::SliderFloat("Pitch Rotation", &rotationX, -1.5f, 1.5f);

        ImGui::Spacing();
        ImGui::Text("Performance: %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
        ImGui::End();

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    // Safely deallocate resources
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &quadVAO);
    glDeleteBuffers(1, &quadVBO);
    glDeleteFramebuffers(1, &fbo);

    glfwTerminate();
    return 0;
}
