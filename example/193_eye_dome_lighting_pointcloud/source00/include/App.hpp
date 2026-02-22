#pragma once
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <vector>
#include <memory>
#include <string>

#include "ArcballCamera.hpp"
#include "GLProgram.hpp"
#include "PostProcessingFBO.hpp"

// ============================================================================
// MAIN APPLICATION ENGINE
// ============================================================================

class App {
public:
    App();
    ~App();

    void run();

private:
    GLFWwindow* window = nullptr;
    ArcballCamera camera;

    std::unique_ptr<GLProgram> geomProgram;
    std::unique_ptr<GLProgram> edlProgram;
    std::unique_ptr<PostProcessingFBO> fbo;

    GLuint vao{0}, vbo{0};
    GLuint quadVAO{0}, quadVBO{0};
    std::vector<glm::vec3> points;

    // Configurable Runtime Parameters
    float edlStrength = 0.8f;
    float edlRadius = 1.5f;
    float edlOffset = 0.001f;
    float pointSize = 35.0f;
    glm::vec3 baseColor = glm::vec3(0.7f, 0.8f, 0.9f);

    void initGLFW();
    void initGLAD();
    void initImGui();
    void loadGeometry();
    [[nodiscard]] std::vector<glm::vec3> loadPointCloud(const std::string& filepath, glm::vec3& outCenter);
    void setupBuffers();
    void renderGeometryPass(int width, int height);
    void renderPostProcessingPass(int width, int height);
    void renderUI();
};
