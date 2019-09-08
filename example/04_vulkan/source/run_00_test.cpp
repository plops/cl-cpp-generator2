// https://vulkan-tutorial.com/
// g++ -std=c++17 run_00_test.cpp  `pkg-config --static --libs glfw3` -lvulkan
// -o run_00_test

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
;
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE

#include <glm/mat4x4.hpp>
#include <glm/vec4.hpp>
;

#include <iostream>
int main() {
  glfwInit();
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  auto window = glfwCreateWindow(800, 600, "vulkan window", nullptr, nullptr);
  uint32_t extensionCount = 0;
  vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
  glm::mat4 matrix;
  glm::vec4 vec;
  auto test = ((matrix) * (vec));
  while (!(glfwWindowShouldClose(window))) {
    glfwPollEvents();
  }
  glfwDestroyWindow(window);
  glfwTerminate();
  return 0;
}