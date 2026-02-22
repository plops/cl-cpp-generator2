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

const auto geometryVS = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
uniform mat4 mvp;

void main() {
    gl_Position = mvp * vec4(aPos, 1.0);
}
)";

const auto geometryFS = R"(
#version 330 core
layout (location = 0) out vec4 FragColor;
uniform vec3 baseColor;

void main() {
    FragColor = vec4(baseColor, 1.0);
}
)";

const auto quadVS = R"(
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aTexCoords;

out vec2 TexCoords;

void main() {
    TexCoords = aTexCoords;
    gl_Position = vec4(aPos.x, aPos.y, 0.0, 1.0);
}
)";

// [NEW] Bilateral Depth-Filling Filter
const auto filterFS = R"(
#version 330 core
in vec2 TexCoords;

uniform sampler2D depthTexture;
uniform int u_filterRadius;
uniform float u_depthThreshold;
uniform float u_sigmaSpatial;
uniform float zNear;
uniform float zFar;

float linearizeDepth(float depth) {
    float z = depth * 2.0 - 1.0;
    return (2.0 * zNear * zFar) / (zFar + zNear - z * (zFar - zNear));
}

void main() {
    float centerRawDepth = texture(depthTexture, TexCoords).r;

    if (u_filterRadius == 0) {
        gl_FragDepth = centerRawDepth;
        return;
    }

    vec2 texSize = vec2(textureSize(depthTexture, 0));
    vec2 texelSize = 1.0 / texSize;

    bool isHole = (centerRawDepth >= 1.0);
    float centerLinearDepth = isHole ? 0.0 : linearizeDepth(centerRawDepth);

    float bestHoleRawDepth = 1.0;
    float minHoleDist = 9999.0;

    float sumWeight = 0.0;
    float sumRawDepth = 0.0;

    for (int y = -u_filterRadius; y <= u_filterRadius; ++y) {
        for (int x = -u_filterRadius; x <= u_filterRadius; ++x) {
            vec2 offset = vec2(float(x), float(y));
            vec2 sampleCoords = TexCoords + offset * texelSize;
            float sampleRawDepth = texture(depthTexture, sampleCoords).r;

            if (sampleRawDepth >= 1.0) continue;

            if (isHole) {
                float distSq = dot(offset, offset);
                if (distSq < minHoleDist) {
                    minHoleDist = distSq;
                    bestHoleRawDepth = sampleRawDepth;
                }
            } else {
                float spatialDistSq = dot(offset, offset);
                float spatialWeight = exp(-spatialDistSq / (2.0 * u_sigmaSpatial * u_sigmaSpatial));

                float sampleLinearDepth = linearizeDepth(sampleRawDepth);
                float depthDiff = abs(sampleLinearDepth - centerLinearDepth);
                float depthWeight = (depthDiff < u_depthThreshold) ? 1.0 : 0.0;

                float weight = spatialWeight * depthWeight;
                sumWeight += weight;
                sumRawDepth += sampleRawDepth * weight;
            }
        }
    }

    if (isHole) {
        gl_FragDepth = bestHoleRawDepth;
    } else {
        gl_FragDepth = (sumWeight > 0.0) ? (sumRawDepth / sumWeight) : centerRawDepth;
    }
}
)";

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

float linearizeDepth(float depth) {
    float z = depth * 2.0 - 1.0;
    return (2.0 * zNear * zFar) / (zFar + zNear - z * (zFar - zNear));
}

void main() {
    float rawDepth = texture(depthTexture, TexCoords).r;

    if (rawDepth >= 1.0) {
        FragColor = vec4(0.15, 0.15, 0.15, 1.0);
        return;
    }

    vec4 baseColor = texture(colorTexture, TexCoords);
    float depth = linearizeDepth(rawDepth);
    vec2 texelSize = 1.0 / resolution;

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

        if (neighborRawDepth >= 1.0) {
            sum += 10.0;
        } else {
            float neighborDepth = linearizeDepth(neighborRawDepth);
            sum += max(0.0, depth - neighborDepth - edlOffset);
        }
    }

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

auto loadPointCloud(const std::string& filepath, glm::vec3& outCenter) -> std::vector<glm::vec3> {
    auto points = std::vector<glm::vec3>{};
    auto file = std::ifstream{filepath};

    if (!file.is_open()) {
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
    if (!glfwInit()) return -1;

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    auto window = glfwCreateWindow(1280, 720, "EDL + Bilateral Filter Viewer", nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress))) return -1;

    auto center = glm::vec3(0.0f);
    auto points = loadPointCloud("../data/bunny.xyz", center);

    if(points.empty()) {
        for(auto i = 0; i < 10000; i++) {
            points.push_back(glm::vec3(
                (rand() % 100) / 10.0f - 5.0f,
                (rand() % 100) / 10.0f - 5.0f,
                (rand() % 100) / 10.0f - 5.0f
            ));
        }
    } else {
        for(auto& p : points) p -= center;
    }

    auto vao = GLuint{}, vbo = GLuint{};
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, points.size() * sizeof(glm::vec3), points.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), nullptr);
    glEnableVertexAttribArray(0);

    float quadVertices[] = {
        -1.0f,  1.0f,  0.0f, 1.0f,
        -1.0f, -1.0f,  0.0f, 0.0f,
         1.0f, -1.0f,  1.0f, 0.0f,
        -1.0f,  1.0f,  0.0f, 1.0f,
         1.0f, -1.0f,  1.0f, 0.0f,
         1.0f,  1.0f,  1.0f, 1.0f
    };

    auto quadVAO = GLuint{}, quadVBO = GLuint{};
    glGenVertexArrays(1, &quadVAO);
    glGenBuffers(1, &quadVBO);
    glBindVertexArray(quadVAO);
    glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), reinterpret_cast<void*>(0));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), reinterpret_cast<void*>(2 * sizeof(float)));

    // 5a. Geometry FBO
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
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, 1280, 720, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthTex, 0);

    // 5b. Bilateral Filter FBO (Writes only to depth)
    auto filterFbo = GLuint{};
    glGenFramebuffers(1, &filterFbo);
    glBindFramebuffer(GL_FRAMEBUFFER, filterFbo);

    auto filteredDepthTex = GLuint{};
    glGenTextures(1, &filteredDepthTex);
    glBindTexture(GL_TEXTURE_2D, filteredDepthTex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, 1280, 720, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, filteredDepthTex, 0);
    glDrawBuffer(GL_NONE); // We are only outputting to gl_FragDepth
    glReadBuffer(GL_NONE);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    auto geomProgram = createProgram(geometryVS, geometryFS);
    auto filterProgram = createProgram(quadVS, filterFS); // [NEW]
    auto edlProgram = createProgram(quadVS, edlFS);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330 core");

    auto edlStrength = 0.8f;
    auto edlRadius = 1.5f;
    auto edlOffset = 0.001f;
    auto pointSize = 3.0f;
    auto baseColor = glm::vec3(0.7f, 0.8f, 0.9f);

    // [NEW] Filter configuration
    int filterMode = 0;
    const char* filterNames[] = { "Off", "3x3 Kernel", "5x5 Kernel", "7x7 Kernel" };
    float depthThreshold = 0.2f;
    float sigmaSpatial = 2.0f;
    bool fKeyWasPressed = false;

    auto cameraDist = 20.0f;
    auto rotationY = 0.0f;
    auto rotationX = 0.5f;

    auto zNear = 0.1f;
    auto zFar = 1000.0f;

    glEnable(GL_DEPTH_TEST);

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        // Check for 'F' key press to cycle stages
        bool fKeyIsPressed = (glfwGetKey(window, GLFW_KEY_F) == GLFW_PRESS);
        if (fKeyIsPressed && !fKeyWasPressed) {
            filterMode = (filterMode + 1) % 4;
        }
        fKeyWasPressed = fKeyIsPressed;

        auto width = 0, height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        if(width == 0 || height == 0) continue;

        glViewport(0, 0, width, height);

        glBindTexture(GL_TEXTURE_2D, colorTex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
        glBindTexture(GL_TEXTURE_2D, depthTex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
        glBindTexture(GL_TEXTURE_2D, filteredDepthTex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);

        // ====================================================================
        // PASS 1: Geometry Rendering
        // ====================================================================
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
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
        // PASS 2: Bilateral Depth Filter
        // ====================================================================
        glBindFramebuffer(GL_FRAMEBUFFER, filterFbo);
        glClear(GL_DEPTH_BUFFER_BIT);
        glUseProgram(filterProgram);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, depthTex);
        glUniform1i(glGetUniformLocation(filterProgram, "depthTexture"), 0);

        glUniform1i(glGetUniformLocation(filterProgram, "u_filterRadius"), filterMode);
        glUniform1f(glGetUniformLocation(filterProgram, "u_depthThreshold"), depthThreshold);
        glUniform1f(glGetUniformLocation(filterProgram, "u_sigmaSpatial"), sigmaSpatial);
        glUniform1f(glGetUniformLocation(filterProgram, "zNear"), zNear);
        glUniform1f(glGetUniformLocation(filterProgram, "zFar"), zFar);

        glBindVertexArray(quadVAO);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        // ====================================================================
        // PASS 3: Eye-Dome Lighting
        // ====================================================================
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glUseProgram(edlProgram);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, colorTex);
        glUniform1i(glGetUniformLocation(edlProgram, "colorTexture"), 0);

        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, filteredDepthTex); // Use our new filtered depth!
        glUniform1i(glGetUniformLocation(edlProgram, "depthTexture"), 1);

        glUniform1f(glGetUniformLocation(edlProgram, "edlStrength"), edlStrength);
        glUniform1f(glGetUniformLocation(edlProgram, "edlRadius"), edlRadius);
        glUniform1f(glGetUniformLocation(edlProgram, "edlOffset"), edlOffset);
        glUniform2f(glGetUniformLocation(edlProgram, "resolution"), static_cast<float>(width), static_cast<float>(height));
        glUniform1f(glGetUniformLocation(edlProgram, "zNear"), zNear);
        glUniform1f(glGetUniformLocation(edlProgram, "zFar"), zFar);

        glBindVertexArray(quadVAO);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        // ====================================================================
        // PASS 4: UI
        // ====================================================================
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::Begin("Pipeline Configuration");
        ImGui::Text("Bilateral Depth Filter");
        ImGui::Separator();
        ImGui::Combo("Filter Stage (F)", &filterMode, filterNames, IM_ARRAYSIZE(filterNames));
        if (filterMode > 0) {
            ImGui::SliderFloat("Depth Tolerance", &depthThreshold, 0.01f, 5.0f, "%.2f");
            ImGui::SliderFloat("Spatial Sigma", &sigmaSpatial, 0.1f, 10.0f, "%.1f");
        }

        ImGui::Spacing();
        ImGui::Text("Non-Photorealistic Shading Settings");
        ImGui::Separator();
        ImGui::SliderFloat("EDL Strength", &edlStrength, 0.0f, 3.0f, "%.2f");
        ImGui::SliderFloat("EDL Radius", &edlRadius, 0.0f, 10.0f, "%.1f px");
        ImGui::SliderFloat("EDL Offset", &edlOffset, 0.0f, 0.01f, "%.4f");

        ImGui::Spacing();
        ImGui::Text("Camera Settings");
        ImGui::Separator();
        ImGui::SliderFloat("Zoom Distance", &cameraDist, 0.1f, 200.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("Yaw Rotation", &rotationY, 0.0f, 6.28f);
        ImGui::SliderFloat("Pitch Rotation", &rotationX, -1.5f, 1.5f);

        ImGui::End();

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &quadVAO);
    glDeleteBuffers(1, &quadVBO);
    glDeleteFramebuffers(1, &fbo);
    glDeleteFramebuffers(1, &filterFbo);

    glfwTerminate();
    return 0;
}