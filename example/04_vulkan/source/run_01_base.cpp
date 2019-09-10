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

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <optional>
#include <set>
#include <stdexcept>
// code to load binary shader from file
#include <fstream>
std::vector<char> readFile(const std::string &filename) {
  auto file = std::ifstream(filename, ((std::ios::ate) | (std::ios::binary)));
  if (!(file.is_open())) {
    throw std::runtime_error("failed to open file.");
  };
  auto fileSize = file.tellg();
  auto buffer = std::vector<char>(fileSize);
  file.seekg(0);
  file.read(buffer.data(), fileSize);
  file.close();
  return buffer;
};
struct QueueFamilyIndices {
  std::optional<uint32_t> graphicsFamily;
  std::optional<uint32_t> presentFamily;
};
typedef struct QueueFamilyIndices QueueFamilyIndices;
struct SwapChainSupportDetails {
  VkSurfaceCapabilitiesKHR capabilities;
  std::vector<VkSurfaceFormatKHR> formats;
  std::vector<VkPresentModeKHR> presentModes;
};
typedef struct SwapChainSupportDetails SwapChainSupportDetails;
SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device,
                                              VkSurfaceKHR surface) {
  SwapChainSupportDetails details;
  vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface,
                                            &details.capabilities);
  uint32_t formatCount = 0;
  vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);
  if (!((0) == (formatCount))) {
    details.formats.resize(formatCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount,
                                         details.formats.data());
  };
  uint32_t presentModeCount = 0;
  vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount,
                                            nullptr);
  if (!((0) == (presentModeCount))) {
    details.presentModes.resize(presentModeCount);
    vkGetPhysicalDeviceSurfacePresentModesKHR(
        device, surface, &presentModeCount, details.presentModes.data());
  };
  return details;
}
VkSurfaceFormatKHR chooseSwapSurfaceFormat(
    const std::vector<VkSurfaceFormatKHR> &availableFormats) {
  for (auto &format : availableFormats) {
    if ((((VK_FORMAT_B8G8R8A8_UNORM) == (format.format)) &&
         ((VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) == (format.colorSpace)))) {
      return format;
    };
  };
  return availableFormats[0];
}
VkPresentModeKHR
chooseSwapPresentMode(const std::vector<VkPresentModeKHR> &modes) {
  // prefer triple buffer (if available)
  for (auto &mode : modes) {
    if ((VK_PRESENT_MODE_MAILBOX_KHR) == (mode)) {
      return mode;
    };
  };
  return VK_PRESENT_MODE_FIFO_KHR;
}
VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR &capabilities) {
  if ((UINT32_MAX) != (capabilities.currentExtent.width)) {
    return capabilities.currentExtent;
  } else {
    VkExtent2D actualExtent = {800, 600};
    actualExtent.width = std::max(
        capabilities.minImageExtent.width,
        std::min(capabilities.maxImageExtent.width, actualExtent.width));
    actualExtent.height = std::max(
        capabilities.minImageExtent.height,
        std::min(capabilities.maxImageExtent.height, actualExtent.height));
    return actualExtent;
  }
};
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
  const std::vector<const char *> _deviceExtensions = {
      VK_KHR_SWAPCHAIN_EXTENSION_NAME};
  VkQueue _presentQueue;
  VkSurfaceKHR _surface;
  VkSwapchainKHR _swapChain;
  std::vector<VkImage> _swapChainImages;
  VkFormat _swapChainImageFormat;
  VkExtent2D _swapChainExtent;
  std::vector<VkImageView> _swapChainImageViews;
  QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) {
    QueueFamilyIndices indices;
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount,
                                             nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount,
                                             queueFamilies.data());
    auto i = 0;
    for (auto &family : queueFamilies) {
      if (((0 < family.queueCount) &&
           (((family.queueFlags) & (VK_QUEUE_GRAPHICS_BIT))))) {
        indices.graphicsFamily = i;
      };
      VkBool32 presentSupport = false;
      vkGetPhysicalDeviceSurfaceSupportKHR(device, i, _surface,
                                           &presentSupport);
      if (((0 < family.queueCount) && (presentSupport))) {
        indices.presentFamily = i;
      };
      (i)++;
    };
    return indices;
  }
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
    // initialize member _instance
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
    // create window surface because it can influence physical device selection
    createSurface();
    pickPhysicalDevice();
    createLogicalDevice();
    createSwapChain();
    createImageViews();
    createGraphicsPipeline();
  }
  // shader stuff
  void createGraphicsPipeline() {
    auto vertShaderModule = createShaderModule(readFile("vert.spv"));
    auto fragShaderModule = createShaderModule(readFile("frag.spv"));
    VkPipelineShaderStageCreateInfo fragShaderStageInfo = {};
    fragShaderStageInfo.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragShaderStageInfo.module = fragShaderModule;
    fragShaderStageInfo.pName = "main";
    fragShaderStageInfo.pSpecializationInfo = nullptr;
    VkPipelineShaderStageCreateInfo vertShaderStageInfo = {};
    vertShaderStageInfo.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertShaderStageInfo.module = vertShaderModule;
    vertShaderStageInfo.pName = "main";
    vertShaderStageInfo.pSpecializationInfo = nullptr;
    VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo,
                                                      fragShaderStageInfo};
    vkDestroyShaderModule(_device, fragShaderModule, nullptr);
    vkDestroyShaderModule(_device, vertShaderModule, nullptr);
  }
  VkShaderModule createShaderModule(const std::vector<char> &code) {
    VkShaderModuleCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t *>(code.data());
    VkShaderModule shaderModule;
    if (!((VK_SUCCESS) == (vkCreateShaderModule(_device, &createInfo, nullptr,
                                                &shaderModule)))) {
      throw std::runtime_error("failed to create shader module.");
    };
    return shaderModule;
  };
  void createSurface() {
    // initialize _surface member
    // must be destroyed before the instance is destroyed
    if (!((VK_SUCCESS) ==
          (glfwCreateWindowSurface(_instance, _window, nullptr, &_surface)))) {
      throw std::runtime_error("failed to create window surface");
    };
  }
  void createSwapChain() {
    auto swapChainSupport = querySwapChainSupport(_physicalDevice, _surface);
    auto surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
    auto presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
    auto extent = chooseSwapExtent(swapChainSupport.capabilities);
    auto imageCount = ((swapChainSupport.capabilities.minImageCount) + (1));
    auto indices = findQueueFamilies(_physicalDevice);
    auto queueFamilyIndices = {indices.graphicsFamily.value(),
                               indices.presentFamily.value()};
    auto imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    auto queueFamilyIndexCount = 0;
    auto pQueueFamilyIndices = nullptr;
    if (!((indices.presentFamily) == (indices.graphicsFamily))) {
      // this could be improved with ownership stuff
      imageSharingMode = VK_SHARING_MODE_CONCURRENT;
      queueFamilyIndexCount = 2;
      pQueueFamilyIndices = pQueueFamilyIndices;
    };
    if (((0 < swapChainSupport.capabilities.maxImageCount) &&
         (swapChainSupport.capabilities.maxImageCount < imageCount))) {
      imageCount = swapChainSupport.capabilities.maxImageCount;
    };
    VkSwapchainCreateInfoKHR createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface = _surface;
    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = surfaceFormat.format;
    createInfo.imageColorSpace = surfaceFormat.colorSpace;
    createInfo.imageExtent = extent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    createInfo.imageSharingMode = imageSharingMode;
    createInfo.queueFamilyIndexCount = queueFamilyIndexCount;
    createInfo.pQueueFamilyIndices = pQueueFamilyIndices;
    createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    createInfo.presentMode = presentMode;
    createInfo.clipped = VK_TRUE;
    createInfo.oldSwapchain = VK_NULL_HANDLE;
    if (!((VK_SUCCESS) ==
          (vkCreateSwapchainKHR(_device, &createInfo, nullptr, &_swapChain)))) {
      throw std::runtime_error("failed to create swap chain");
    };
    // now get the images, note will be destroyed with the swap chain
    vkGetSwapchainImagesKHR(_device, _swapChain, &imageCount, nullptr);
    _swapChainImages.resize(imageCount);
    vkGetSwapchainImagesKHR(_device, _swapChain, &imageCount,
                            _swapChainImages.data());
    _swapChainImageFormat = surfaceFormat.format;
    _swapChainExtent = extent;
  }
  void createImageViews() {
    _swapChainImageViews.resize(_swapChainImages.size());
    for (int i = 0; i < _swapChainImages.size(); (i) += (1)) {
      VkImageViewCreateInfo createInfo = {};
      createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
      createInfo.image = _swapChainImages[i];
      createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
      createInfo.format = _swapChainImageFormat;
      createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
      createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
      createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
      createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
      createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      createInfo.subresourceRange.baseMipLevel = 0;
      createInfo.subresourceRange.levelCount = 1;
      createInfo.subresourceRange.baseArrayLayer = 0;
      createInfo.subresourceRange.layerCount = 1;
      if (!((VK_SUCCESS) == (vkCreateImageView(_device, &createInfo, nullptr,
                                               &(_swapChainImageViews[i]))))) {
        throw std::runtime_error("failed to create image view.");
      };
    }
  };
  void createLogicalDevice() {
    // initialize members _device and _graphicsQueue
    auto indices = findQueueFamilies(_physicalDevice);
    float queuePriority = (1.e+0);
    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    std::set<uint32_t> uniqueQueueFamilies = {indices.graphicsFamily.value(),
                                              indices.presentFamily.value()};
    for (auto &queueFamily : uniqueQueueFamilies) {
      VkDeviceQueueCreateInfo queueCreateInfo = {};
      queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
      queueCreateInfo.queueFamilyIndex = queueFamily;
      queueCreateInfo.queueCount = 1;
      queueCreateInfo.pQueuePriorities = &queuePriority;
      queueCreateInfos.push_back(queueCreateInfo);
    };
    VkPhysicalDeviceFeatures deviceFeatures = {};
    VkDeviceCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.pQueueCreateInfos = queueCreateInfos.data();
    createInfo.queueCreateInfoCount =
        static_cast<uint32_t>(queueCreateInfos.size());
    createInfo.pEnabledFeatures = &deviceFeatures;
    createInfo.enabledExtensionCount =
        static_cast<uint32_t>(_deviceExtensions.size());
    createInfo.ppEnabledExtensionNames = _deviceExtensions.data();
    createInfo.enabledLayerCount =
        static_cast<uint32_t>(_validationLayers.size());
    createInfo.ppEnabledLayerNames = _validationLayers.data();
    if (!((VK_SUCCESS) ==
          (vkCreateDevice(_physicalDevice, &createInfo, nullptr, &_device)))) {
      throw std::runtime_error("failed to create logical device");
    };
    vkGetDeviceQueue(_device, indices.graphicsFamily.value(), 0,
                     &_graphicsQueue);
    vkGetDeviceQueue(_device, indices.presentFamily.value(), 0, &_presentQueue);
  }
  bool checkDeviceExtensionSupport(VkPhysicalDevice device) {
    uint32_t extensionCount = 0;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount,
                                         nullptr);
    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount,
                                         availableExtensions.data());
    std::set<std::string> requiredExtensions(_deviceExtensions.begin(),
                                             _deviceExtensions.end());
    for (auto &extension : availableExtensions) {
      requiredExtensions.erase(extension.extensionName);
    };
    return requiredExtensions.empty();
  }
  bool isDeviceSuitable(VkPhysicalDevice device) {
    auto extensionsSupported = checkDeviceExtensionSupport(device);
    bool swapChainAdequate = false;
    if (extensionsSupported) {
      auto swapChainSupport = querySwapChainSupport(device, _surface);
      swapChainAdequate = ((!(swapChainSupport.formats.empty())) &&
                           (!(swapChainSupport.presentModes.empty())));
    };
    QueueFamilyIndices indices = findQueueFamilies(device);
    return ((indices.graphicsFamily.has_value()) &&
            (((indices.presentFamily.has_value()) && (extensionsSupported) &&
              (swapChainAdequate))));
  }
  void pickPhysicalDevice() {
    // initialize member _physicalDevice
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
    for (auto &view : _swapChainImageViews) {
      vkDestroyImageView(_device, view, nullptr);
    };
    vkDestroySwapchainKHR(_device, _swapChain, nullptr);
    vkDestroyDevice(_device, nullptr);
    vkDestroySurfaceKHR(_instance, _surface, nullptr);
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