#include "App.hpp"
#include "Shaders.hpp"
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <limits>
#include <stdexcept>

App::App() {
    initGLFW();
    initGLAD();
    initImGui();
    loadGeometry();
    setupBuffers();

    geomProgram = std::make_unique<GLProgram>(Shaders::geometryVS, Shaders::geometryFS);
    edlProgram = std::make_unique<GLProgram>(Shaders::edlVS, Shaders::edlFS);
    fbo = std::make_unique<PostProcessingFBO>();
}

App::~App() {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &quadVAO);
    glDeleteBuffers(1, &quadVBO);

    glfwDestroyWindow(window);
    glfwTerminate();
}

void App::run() {
    glEnable(GL_DEPTH_TEST);

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        camera.processMouseInput(window);

        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        if (width == 0 || height == 0) continue;

        glViewport(0, 0, width, height);
        fbo->resizeIfNeeded(width, height);

        renderGeometryPass(width, height);
        renderPostProcessingPass(width, height);
        renderUI();

        glfwSwapBuffers(window);
    }
}

void App::initGLFW() {
    if (!glfwInit()) {
        throw std::runtime_error("Failed to initialize GLFW");
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    window = glfwCreateWindow(1280, 720, "Eye-Dome Lighting Point Cloud Viewer", nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        throw std::runtime_error("Failed to create GLFW window");
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable V-Sync
}

void App::initGLAD() {
    if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress))) {
        throw std::runtime_error("Failed to initialize GLAD");
    }
}

void App::initImGui() {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330 core");
}

void App::loadGeometry() {
    auto center = glm::vec3(0.0f);
    points = loadPointCloud("../data/bunny.xyz", center);

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
}

std::vector<glm::vec3> App::loadPointCloud(const std::string& filepath, glm::vec3& outCenter) {
    auto pts = std::vector<glm::vec3>{};
    auto file = std::ifstream{filepath};

    if (!file.is_open()) {
        std::cerr << "Warning: Failed to open point cloud file: " << filepath << std::endl;
        return pts;
    }

    auto line = std::string{};
    auto minBounds = glm::vec3(std::numeric_limits<float>::max());
    auto maxBounds = glm::vec3(std::numeric_limits<float>::lowest());

    while (std::getline(file, line)) {
        auto ss = std::istringstream{line};
        auto p = glm::vec3{};
        if (ss >> p.x >> p.y >> p.z) {
            pts.push_back(p);
            minBounds = glm::min(minBounds, p);
            maxBounds = glm::max(maxBounds, p);
        }
    }

    outCenter = (minBounds + maxBounds) * 0.5f;
    return pts;
}

void App::setupBuffers() {
    // Setup Geometry VAO/VBO for the Point Cloud
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

    glGenVertexArrays(1, &quadVAO);
    glGenBuffers(1, &quadVBO);

    glBindVertexArray(quadVAO);
    glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), reinterpret_cast<void*>(0));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), reinterpret_cast<void*>(2 * sizeof(float)));
}

void App::renderGeometryPass(int width, int height) {
    glBindFramebuffer(GL_FRAMEBUFFER, fbo->fbo);
    // Clear color to pure white representing infinite depth space (1.0)
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    geomProgram->use();

    auto proj = glm::perspective(glm::radians(60.0f), static_cast<float>(width)/height, camera.zNear, camera.zFar);
    auto mvp = proj * camera.getViewMatrix();

    geomProgram->setUniform("mvp", mvp);
    geomProgram->setUniform("baseColor", baseColor);

    glPointSize(pointSize);
    glBindVertexArray(vao);
    glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(points.size()));
}

void App::renderPostProcessingPass(int width, int height) {
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    edlProgram->use();

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, fbo->colorTex);
    edlProgram->setUniform("colorTexture", 0);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, fbo->depthTex);
    edlProgram->setUniform("depthTexture", 1);

    // Push ImGui tuned variables to the GPU
    edlProgram->setUniform("edlStrength", edlStrength);
    edlProgram->setUniform("edlRadius", edlRadius);
    edlProgram->setUniform("edlOffset", edlOffset);
    edlProgram->setUniform("resolution", glm::vec2(width, height));
    edlProgram->setUniform("zNear", camera.zNear);
    edlProgram->setUniform("zFar", camera.zFar);

    glBindVertexArray(quadVAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);
}

void App::renderUI() {
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
    ImGui::SliderFloat("Point Size", &pointSize, 1.0f, 35.0f, "%.1f px");
    ImGui::ColorEdit3("Base Color", glm::value_ptr(baseColor));

    ImGui::Spacing();
    ImGui::Text("Camera Settings");
    ImGui::Separator();
    ImGui::SliderFloat("Zoom Distance", &camera.distance, 0.1f, 200.0f, "%.3f", ImGuiSliderFlags_Logarithmic);
    ImGui::SliderFloat("Yaw Rotation", &camera.rotationY, 0.0f, 6.28f);
    ImGui::SliderFloat("Pitch Rotation", &camera.rotationX, -1.5f, 1.5f);
    ImGui::TextDisabled("(You can also click and drag the mouse to rotate)");

    ImGui::Spacing();
    ImGui::Text("Performance: %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
    ImGui::End();

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}
