#include "ArcballCamera.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include <imgui.h>
#include <algorithm>

void ArcballCamera::processMouseInput(GLFWwindow* window) {
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

glm::mat4 ArcballCamera::getViewMatrix() const {
    auto view = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, -distance));
    view = glm::rotate(view, rotationX, glm::vec3(1.0f, 0.0f, 0.0f));
    view = glm::rotate(view, rotationY, glm::vec3(0.0f, 1.0f, 0.0f));
    return view;
}
