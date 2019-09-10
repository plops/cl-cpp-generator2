// https://vulkan-tutorial.com/en/Drawing_a_triangle/Setup/Base_code
// https://vulkan-tutorial.com/en/Drawing_a_triangle/Setup/Validation_layers
// g++ -std=c++17 run_01_base.cpp  `pkg-config --static --libs glfw3` -lvulkan
// -o run_01_base

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
;
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE

#include <glm/mat4x4.hpp>
#include <glm/vec4.hpp>
;

#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <optional>
#include <stdexcept>
struct QueueFamilyIndices {
  std::optional<uint32_t> graphicsFamily;
};
typedef struct QueueFamilyIndices QueueFamilyIndices;
QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) {
  QueueFamilyIndices indices;
  uint32_t queueFamilyCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);
  std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
  vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount,
                                           queueFamilies.data());
  auto i = 0;
  for (auto &family : queueFamilies) {
    if (((0 < family.queueCount) &&
         (((family.queueFlags) & (VK_QUEUE_GRAPHICS_BIT))))) {
      indices.graphicsFamily = i;
    };
    (i)++;
  };
  return indices;
}
class HelloTriangleApplication {
public:
  void run() {
    initWindow();
    initVulkan();
    mainLoop();
    cleanup();
  }

private:
  GLFWwindow *_window;
  VkInstance _instance;
  const std::vector<const char *> _validationLayers = {
      "VK_LAYER_KHRONOS_validation"};
  VkPhysicalDevice _physicalDevice = VK_NULL_HANDLE;
  VkDevice _device;
  VkQueue _graphicsQueue;
  bool checkValidationLayerSupport() {
    uint32_t layerCount = 0;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
    std::vector<VkLayerProperties> availableLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());
    for (auto &layerName : _validationLayers) {
      auto layerFound = false;
      for (auto &layerProperties : availableLayers) {
        if ((0) == (strcmp(layerName, layerProperties.layerName))) {
          layerFound = true;
          break;
        };
      };
      if (!(layerFound)) {
        return false;
      };
    };
    return true;
  }
  void initWindow() {
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    _window = glfwCreateWindow(800, 600, "vulkan window", nullptr, nullptr);
  }
  void createInstance() {
#initialize member _instance
    if (!(checkValidationLayerSupport())) {
      throw std::runtime_error("validation layers requested, but unavailable.");
    };
    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Hello Triangle";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_0;
    uint32_t glfwExtensionCount = 0;
    const char **glfwExtensions;
    glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
    VkInstanceCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
    createInfo.enabledExtensionCount = glfwExtensionCount;
    createInfo.ppEnabledExtensionNames = glfwExtensions;
    createInfo.enabledLayerCount =
        static_cast<uint32_t>(_validationLayers.size());
    createInfo.ppEnabledLayerNames = _validationLayers.data();
    if (!((VK_SUCCESS) ==
          (vkCreateInstance(&createInfo, nullptr, &_instance)))) {
      throw std::runtime_error("failed to create instance");
    };
  }
  void initVulkan() {
    createInstance();
    pickPhysicalDevice();
    createLogicalDevice();
  }
  void createLogicalDevice() {
#initialize members _device and _graphicsQueue
    auto indices = findQueueFamilies(_physicalDevice);
    float queuePriority = (1.e+0);
    VkDeviceQueueCreateInfo queueCreateInfo = {};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = indices.graphicsFamily.value();
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = &queuePriority;
    VkPhysicalDeviceFeatures deviceFeatures = {};
    VkDeviceCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.pQueueCreateInfos = &queueCreateInfo;
    createInfo.queueCreateInfoCount = 1;
    createInfo.pEnabledFeatures = &deviceFeatures;
    createInfo.enabledExtensionCount = 0;
    createInfo.enabledLayerCount =
        static_cast<uint32_t>(_validationLayers.size());
    createInfo.ppEnabledLayerNames = _validationLayers.data();
    if (!((VK_SUCCESS) ==
          (vkCreateDevice(_physicalDevice, &createInfo, nullptr, &_device)))) {
      throw std::runtime_error("failed to create logical device");
    };
  }
  bool isDeviceSuitable(VkPhysicalDevice device) {
    QueueFamilyIndices indices = findQueueFamilies(device);
    return indices.graphicsFamily.has_value();
  }
  void pickPhysicalDevice() {
#initialize member _physicalDevice
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(_instance, &deviceCount, nullptr);
    if ((0) == (deviceCount)) {
      throw std::runtime_error("failed to find gpu with vulkan support.");
    };
    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(_instance, &deviceCount, devices.data());
    for (auto &device : devices) {
      if (isDeviceSuitable(device)) {
        _physicalDevice = device;
        break;
      };
    };
    if ((VK_NULL_HANDLE) == (_physicalDevice)) {
      throw std::runtime_error("failed to find a suitable gpu.");
    };
  }
  void mainLoop() {
    while (!(glfwWindowShouldClose(_window))) {
      glfwPollEvents();
    }
  }
  void cleanup() {
    vkDestroyDevice(_device, nullptr);
    vkDestroyInstance(_instance, nullptr);
    glfwDestroyWindow(_window);
    glfwTerminate();
  };
};
int main() {
  HelloTriangleApplication app;
  try {
    app.run();
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  };
  return EXIT_SUCCESS;
}