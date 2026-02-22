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
#include <memory>

// ============================================================================
// SHADER SOURCE CODE DEFINITIONS
// ============================================================================
namespace Shaders {
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
}

// ============================================================================
// CORE CLASSES & PATTERNS
// ============================================================================

// RAII Wrapper for OpenGL Shader Programs
class GLProgram {
public:
    GLuint id{0};

    GLProgram(const char* vsSource, const char* fsSource) {
        auto vs = compileShader(GL_VERTEX_SHADER, vsSource);
        auto fs = compileShader(GL_FRAGMENT_SHADER, fsSource);

        id = glCreateProgram();
        glAttachShader(id, vs);
        glAttachShader(id, fs);
        glLinkProgram(id);

        glDeleteShader(vs);
        glDeleteShader(fs);
    }

    ~GLProgram() {
        if (id != 0) glDeleteProgram(id);
    }

    // Delete copy semantics to prevent double-freeing OpenGL resources
    GLProgram(const GLProgram&) = delete;
    GLProgram& operator=(const GLProgram&) = delete;

    void use() const { glUseProgram(id); }

    void setUniform(const char* name, float value) const { glUniform1f(glGetUniformLocation(id, name), value); }
    void setUniform(const char* name, int value) const { glUniform1i(glGetUniformLocation(id, name), value); }
    void setUniform(const char* name, const glm::vec2& value) const { glUniform2fv(glGetUniformLocation(id, name), 1, glm::value_ptr(value)); }
    void setUniform(const char* name, const glm::vec3& value) const { glUniform3fv(glGetUniformLocation(id, name), 1, glm::value_ptr(value)); }
    void setUniform(const char* name, const glm::mat4& value) const { glUniformMatrix4fv(glGetUniformLocation(id, name), 1, GL_FALSE, glm::value_ptr(value)); }

private:
    [[nodiscard]] static GLuint compileShader(GLenum type, const char* source) {
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
};

// Interactive Camera handling Arcball rotation
class ArcballCamera {
public:
    float distance = 0.32f;
    float rotationY = 0.0f;
    float rotationX = 0.5f;

    float zNear = 0.1f;
    float zFar = 1000.0f;

    void processMouseInput(GLFWwindow* window) {
        // Prevent camera rotation if ImGui is being used
        if (ImGui::GetIO().WantCaptureMouse) {
            isDragging = false;
            return;
        }

        double xpos, ypos;
        glfwGetCursorPos(window, &xpos, &ypos);

        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
            if (!isDragging) {
                isDragging = true;
                lastMousePos = glm::vec2(xpos, ypos);
            }

            glm::vec2 delta = glm::vec2(xpos, ypos) - lastMousePos;
            rotationY += delta.x * sensitivity;
            rotationX += delta.y * sensitivity;

            // Optional: Clamp pitch to prevent the camera from flipping upside down
            rotationX = std::clamp(rotationX, -1.5f, 1.5f);

            lastMousePos = glm::vec2(xpos, ypos);
        } else {
            isDragging = false;
        }
    }

    [[nodiscard]] glm::mat4 getViewMatrix() const {
        auto view = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, -distance));
        view = glm::rotate(view, rotationX, glm::vec3(1.0f, 0.0f, 0.0f));
        view = glm::rotate(view, rotationY, glm::vec3(0.0f, 1.0f, 0.0f));
        return view;
    }

private:
    bool isDragging = false;
    glm::vec2 lastMousePos{0.0f, 0.0f};
    const float sensitivity = 0.01f;
};

// RAII Framebuffer Manager
class PostProcessingFBO {
public:
    GLuint fbo{0}, colorTex{0}, depthTex{0};
    int currentWidth = 0, currentHeight = 0;

    PostProcessingFBO() {
        glGenFramebuffers(1, &fbo);
        glGenTextures(1, &colorTex);
        glGenTextures(1, &depthTex);
    }

    ~PostProcessingFBO() {
        glDeleteFramebuffers(1, &fbo);
        glDeleteTextures(1, &colorTex);
        glDeleteTextures(1, &depthTex);
    }

    void resizeIfNeeded(int width, int height) {
        if (width == currentWidth && height == currentHeight) return;
        currentWidth = width;
        currentHeight = height;

        glBindFramebuffer(GL_FRAMEBUFFER, fbo);

        glBindTexture(GL_TEXTURE_2D, colorTex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, colorTex, 0);

        glBindTexture(GL_TEXTURE_2D, depthTex);
        // Crucial: 32-bit float requirement for depth evaluation precision
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthTex, 0);

        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            std::cerr << "Critical Error: Framebuffer architecture incomplete!" << std::endl;
        }
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }
};

// ============================================================================
// MAIN APPLICATION ENGINE
// ============================================================================

class App {
public:
    App() {
        initGLFW();
        initGLAD();
        initImGui();
        loadGeometry();
        setupBuffers();

        geomProgram = std::make_unique<GLProgram>(Shaders::geometryVS, Shaders::geometryFS);
        edlProgram = std::make_unique<GLProgram>(Shaders::edlVS, Shaders::edlFS);
        fbo = std::make_unique<PostProcessingFBO>();
    }

    ~App() {
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

    void run() {
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

    void initGLFW() {
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

    void initGLAD() {
        if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress))) {
            throw std::runtime_error("Failed to initialize GLAD");
        }
    }

    void initImGui() {
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGui_ImplGlfw_InitForOpenGL(window, true);
        ImGui_ImplOpenGL3_Init("#version 330 core");
    }

    void loadGeometry() {
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

    // Parses raw XYZ text data and translates the bounding volume to the origin
    [[nodiscard]] std::vector<glm::vec3> loadPointCloud(const std::string& filepath, glm::vec3& outCenter) {
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

    void setupBuffers() {
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

    void renderGeometryPass(int width, int height) {
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

    void renderPostProcessingPass(int width, int height) {
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

    void renderUI() {
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
};

// ============================================================================
// ENTRY POINT
// ============================================================================

int main() {
    try {
        App app;
        app.run();
    } catch (const std::exception& e) {
        std::cerr << "Fatal Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}