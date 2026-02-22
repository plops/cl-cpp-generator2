#pragma once
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

// Interactive Camera handling Arcball rotation
class ArcballCamera {
public:
    float distance = 0.32f;
    float rotationY = 0.0f;
    float rotationX = 0.5f;

    float zNear = 0.1f;
    float zFar = 1000.0f;

    void processMouseInput(GLFWwindow* window);

    [[nodiscard]] glm::mat4 getViewMatrix() const;

private:
    bool isDragging = false;
    glm::vec2 lastMousePos{0.0f, 0.0f};
    const float sensitivity = 0.01f;
};
