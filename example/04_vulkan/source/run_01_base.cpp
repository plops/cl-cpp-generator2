// https://vulkan-tutorial.com/en/Drawing_a_triangle/Setup/Base_code
// https://vulkan-tutorial.com/en/Drawing_a_triangle/Setup/Validation_layers
// g++ -std=c++17 run_01_base.cpp  `pkg-config --static --libs glfw3` -lvulkan
// -o run_01_base -pedantic -Wall -Wextra -Wcast-align -Wcast-qual
// -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2 -Winit-self
// -Wlogical-op -Wmissing-declarations -Wmissing-include-dirs -Wnoexcept
// -Wold-style-cast -Woverloaded-virtual -Wredundant-decls -Wshadow
// -Wsign-conversion -Wsign-promo -Wstrict-null-sentinel -Wstrict-overflow=5
// -Wswitch-default -Wundef -march=native -O2 -g

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
;
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE

#include <chrono>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
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
typedef struct SwapChainSupportDetails SwapChainSupportDetails;
typedef struct QueueFamilyIndices QueueFamilyIndices;
std::vector<char> readFile(const std::string &);
SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice, VkSurfaceKHR);
VkSurfaceFormatKHR
chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR> &);
VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR> &);
VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR &, GLFWwindow *);
QueueFamilyIndices findQueueFamilies(VkPhysicalDevice, VkSurfaceKHR);
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
struct UniformBufferObject {
  alignas(16) glm::mat4 model;
  alignas(16) glm::mat4 view;
  alignas(16) glm::mat4 proj;
};
typedef struct UniformBufferObject UniformBufferObject;
struct Vertex {
  glm::vec2 pos;
  glm::vec3 color;
  static VkVertexInputBindingDescription getBindingDescription();
  static std::array<VkVertexInputAttributeDescription, 2>
  getAttributeDescriptions();
};
typedef struct Vertex Vertex;
VkVertexInputBindingDescription Vertex::getBindingDescription() {
  VkVertexInputBindingDescription bindingDescription = {};
  bindingDescription.binding = 0;
  bindingDescription.stride = sizeof(Vertex);
  bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
  return bindingDescription;
}
std::vector<Vertex> g_vertices = {
    {{(-5.e-1f), (-5.e-1f)}, {(1.e+0f), (0.0e+0f), (0.0e+0f)}},
    {{(5.e-1f), (-5.e-1f)}, {(0.0e+0f), (1.e+0f), (0.0e+0f)}},
    {{(5.e-1f), (5.e-1f)}, {(0.0e+0f), (0.0e+0f), (1.e+0f)}},
    {{(-5.e-1f), (5.e-1f)}, {(1.e+0f), (1.e+0f), (1.e+0f)}}};
std::vector<uint16_t> g_indices = {0, 1, 2, 2, 3, 0};
std::array<VkVertexInputAttributeDescription, 2>
Vertex::getAttributeDescriptions() {
  std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions = {};
  attributeDescriptions[0].binding = 0;
  attributeDescriptions[0].location = 0;
  attributeDescriptions[0].format = VK_FORMAT_R32G32_SFLOAT;
  attributeDescriptions[0].offset = offsetof(Vertex, pos);
  attributeDescriptions[1].binding = 0;
  attributeDescriptions[1].location = 1;
  attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
  attributeDescriptions[1].offset = offsetof(Vertex, color);
  return attributeDescriptions;
}
struct QueueFamilyIndices {
  std::optional<uint32_t> graphicsFamily;
  std::optional<uint32_t> presentFamily;
  bool isComplete();
};
typedef struct QueueFamilyIndices QueueFamilyIndices;
bool QueueFamilyIndices::isComplete() {
  return ((graphicsFamily.has_value()) && (presentFamily.has_value()));
}
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
VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR &capabilities,
                            GLFWwindow *_window) {
  if ((UINT32_MAX) != (capabilities.currentExtent.width)) {
    return capabilities.currentExtent;
  } else {
    int width = 0;
    int height = 0;
    glfwGetFramebufferSize(_window, &width, &height);
    VkExtent2D actualExtent = {width, height};
    actualExtent.width = std::max(
        capabilities.minImageExtent.width,
        std::min(capabilities.maxImageExtent.width, actualExtent.width));
    actualExtent.height = std::max(
        capabilities.minImageExtent.height,
        std::min(capabilities.maxImageExtent.height, actualExtent.height));
    return actualExtent;
  }
};
bool checkDeviceExtensionSupport(
    VkPhysicalDevice device,
    const std::vector<const char *> _deviceExtensions) {
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
QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device,
                                     VkSurfaceKHR _surface) {
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
    VkBool32 presentSupport = false;
    vkGetPhysicalDeviceSurfaceSupportKHR(device, i, _surface, &presentSupport);
    if (((0 < family.queueCount) && (presentSupport))) {
      indices.presentFamily = i;
    };
    if (indices.isComplete()) {
      break;
    };
    (i)++;
  };
  return indices;
}
bool isDeviceSuitable(VkPhysicalDevice device, VkSurfaceKHR _surface,
                      const std::vector<const char *> _deviceExtensions) {
  auto extensionsSupported =
      checkDeviceExtensionSupport(device, _deviceExtensions);
  bool swapChainAdequate = false;
  if (extensionsSupported) {
    auto swapChainSupport = querySwapChainSupport(device, _surface);
    swapChainAdequate = ((!(swapChainSupport.formats.empty())) &&
                         (!(swapChainSupport.presentModes.empty())));
  };
  QueueFamilyIndices indices = findQueueFamilies(device, _surface);
  return ((indices.isComplete()) &&
          (((extensionsSupported) && (swapChainAdequate))));
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
  VkRenderPass _renderPass;
  VkDescriptorSetLayout _descriptorSetLayout;
  VkPipelineLayout _pipelineLayout;
  VkPipeline _graphicsPipeline;
  std::vector<VkFramebuffer> _swapChainFramebuffers;
  VkCommandPool _commandPool;
  std::vector<VkCommandBuffer> _commandBuffers;
  std::vector<VkSemaphore> _imageAvailableSemaphores;
  std::vector<VkSemaphore> _renderFinishedSemaphores;
  std::vector<VkFence> _inFlightFences;
  const int _MAX_FRAMES_IN_FLIGHT = 2;
  size_t _currentFrame = 0;
  bool _framebufferResized = false;
  VkBuffer _vertexBuffer;
  VkDeviceMemory _vertexBufferMemory;
  VkBuffer _indexBuffer;
  VkDeviceMemory _indexBufferMemory;
  std::vector<VkBuffer> _uniformBuffers;
  std::vector<VkDeviceMemory> _uniformBuffersMemory;
  VkDescriptorPool _descriptorPool;
  std::vector<VkDescriptorSet> _descriptorSets;
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
  static void framebufferResizeCallback(GLFWwindow *window, int width,
                                        int height) {
    auto app = reinterpret_cast<HelloTriangleApplication *>(
        glfwGetWindowUserPointer(window));
    app->_framebufferResized = true;
  }
  void initWindow() {
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    _window = glfwCreateWindow(800, 600, "vulkan window", nullptr, nullptr);
    glfwSetWindowUserPointer(_window, this);
    glfwSetFramebufferSizeCallback(_window, framebufferResizeCallback);
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
      throw std::runtime_error(
          "failed to (vkCreateInstance &createInfo nullptr &_instance)");
    };
  }
  std::tuple<VkBuffer, VkDeviceMemory>
  createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
               VkMemoryPropertyFlags properties) {
    VkBuffer buffer;
    VkDeviceMemory bufferMemory;
    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    bufferInfo.flags = 0;
    if (!((VK_SUCCESS) ==
          (vkCreateBuffer(_device, &bufferInfo, nullptr, &buffer)))) {
      throw std::runtime_error(
          "failed to (vkCreateBuffer _device &bufferInfo nullptr &buffer)");
    };
    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(_device, buffer, &memReq);
    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex =
        findMemoryType(memReq.memoryTypeBits, properties);
    if (!((VK_SUCCESS) ==
          (vkAllocateMemory(_device, &allocInfo, nullptr, &bufferMemory)))) {
      throw std::runtime_error("failed to (vkAllocateMemory _device &allocInfo "
                               "nullptr &bufferMemory)");
    };
    vkBindBufferMemory(_device, buffer, bufferMemory, 0);
    return std::make_tuple(buffer, bufferMemory);
  }
  void createVertexBuffer() {
    auto bufferSize = ((sizeof(g_vertices[0])) * (g_vertices.size()));
    auto [stagingBuffer, stagingBufferMemory] =
        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     ((VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) |
                      (VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)));
    void *data;
    vkMapMemory(_device, stagingBufferMemory, 0, bufferSize, 0, &data);
    memcpy(data, g_vertices.data(), bufferSize);
    vkUnmapMemory(_device, stagingBufferMemory);
    auto [vertexBuffer, vertexBufferMemory] =
        createBuffer(bufferSize,
                     ((VK_BUFFER_USAGE_VERTEX_BUFFER_BIT) |
                      (VK_BUFFER_USAGE_TRANSFER_DST_BIT)),
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    _vertexBuffer = vertexBuffer;
    _vertexBufferMemory = vertexBufferMemory;
    copyBuffer(stagingBuffer, _vertexBuffer, bufferSize);
    vkDestroyBuffer(_device, stagingBuffer, nullptr);
    vkFreeMemory(_device, stagingBufferMemory, nullptr);
  }
  void createIndexBuffer() {
    auto bufferSize = ((sizeof(g_indices[0])) * (g_indices.size()));
    auto [stagingBuffer, stagingBufferMemory] =
        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     ((VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) |
                      (VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)));
    void *data;
    vkMapMemory(_device, stagingBufferMemory, 0, bufferSize, 0, &data);
    memcpy(data, g_indices.data(), bufferSize);
    vkUnmapMemory(_device, stagingBufferMemory);
    auto [indexBuffer, indexBufferMemory] =
        createBuffer(bufferSize,
                     ((VK_BUFFER_USAGE_INDEX_BUFFER_BIT) |
                      (VK_BUFFER_USAGE_TRANSFER_DST_BIT)),
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    _indexBuffer = indexBuffer;
    _indexBufferMemory = indexBufferMemory;
    copyBuffer(stagingBuffer, _indexBuffer, bufferSize);
    vkDestroyBuffer(_device, stagingBuffer, nullptr);
    vkFreeMemory(_device, stagingBufferMemory, nullptr);
  }
  void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
    VkCommandBuffer commandBuffer;
    {
      VkCommandBufferAllocateInfo info = {};
      info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
      info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
      info.commandPool = _commandPool;
      info.commandBufferCount = 1;
      vkAllocateCommandBuffers(_device, &info, &commandBuffer);
    };
    {
      {
        VkCommandBufferBeginInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(commandBuffer, &info);
      };
      VkBufferCopy copyRegion = {};
      copyRegion.srcOffset = 0;
      copyRegion.dstOffset = 0;
      copyRegion.size = size;
      vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);
      vkEndCommandBuffer(commandBuffer);
      VkSubmitInfo submitInfo = {};
      submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
      submitInfo.commandBufferCount = 1;
      submitInfo.pCommandBuffers = &commandBuffer;
      vkQueueSubmit(_graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
      vkQueueWaitIdle(_graphicsQueue);
    };
    vkFreeCommandBuffers(_device, _commandPool, 1, &commandBuffer);
  }
  uint32_t findMemoryType(uint32_t typeFilter,
                          VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties ps;
    vkGetPhysicalDeviceMemoryProperties(_physicalDevice, &ps);
    for (int i = 0; i < ps.memoryTypeCount; (i) += (1)) {
      if (((((1 << i) & (typeFilter))) &&
           ((properties) ==
            (((properties) & (ps.memoryTypes[i].propertyFlags)))))) {
        return i;
      };
    }
    throw std::runtime_error("failed to find suitable memory type.");
  }
  void initVulkan() {
    createInstance();
    // create window surface because it can influence physical device selection
    createSurface();
    pickPhysicalDevice();
    createLogicalDevice();
    createSwapChain();
    createImageViews();
    createRenderPass();
    createDescriptorSetLayout();
    createGraphicsPipeline();
    createFramebuffers();
    createCommandPool();
    createVertexBuffer();
    createIndexBuffer();
    createUniformBuffers();
    createDescriptorPool();
    createDescriptorSets();
    createCommandBuffers();
    createSyncObjects();
  }
  void createDescriptorSets() {
    auto n = static_cast<uint32_t>(_swapChainImages.size());
    std::vector<VkDescriptorSetLayout> layouts(n, _descriptorSetLayout);
    {
      VkDescriptorSetAllocateInfo info = {};
      info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
      info.descriptorPool = _descriptorPool;
      info.descriptorSetCount = n;
      info.pSetLayouts = layouts.data();
      if (!((VK_SUCCESS) == (vkAllocateDescriptorSets(
                                _device, &info, _descriptorSets.data())))) {
        throw std::runtime_error("failed to (vkAllocateDescriptorSets _device "
                                 "&info (_descriptorSets.data))");
      };
    };
    for (int i = 0; i < n; (i) += (1)) {
      VkDescriptorBufferInfo bufferInfo = {};
      bufferInfo.buffer = _uniformBuffers[i];
      bufferInfo.offset = 0;
      bufferInfo.range = sizeof(UniformBufferObject);
      VkWriteDescriptorSet descriptorWrite = {};
      descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      descriptorWrite.dstSet = _descriptorSets[i];
      descriptorWrite.dstBinding = 0;
      descriptorWrite.dstArrayElement = 0;
      descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
      descriptorWrite.descriptorCount = 1;
      descriptorWrite.pBufferInfo = &bufferInfo;
      descriptorWrite.pImageInfo = nullptr;
      descriptorWrite.pTexelBufferView = nullptr;
      vkUpdateDescriptorSets(_device, 1, &descriptorWrite, 0, nullptr);
    };
  }
  void createDescriptorPool() {
    auto n = static_cast<uint32_t>(_swapChainImages.size());
    VkDescriptorPoolSize poolSize = {};
    poolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSize.descriptorCount = n;
    {
      VkDescriptorPoolCreateInfo info = {};
      info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
      info.poolSizeCount = 1;
      info.pPoolSizes = &poolSize;
      info.maxSets = n;
      info.flags = 0;
      if (!((VK_SUCCESS) == (vkCreateDescriptorPool(_device, &info, nullptr,
                                                    &_descriptorPool)))) {
        throw std::runtime_error("failed to (vkCreateDescriptorPool _device "
                                 "&info nullptr &_descriptorPool)");
      };
    };
  }
  void createUniformBuffers() {
    auto bufferSize = sizeof(UniformBufferObject);
    auto n = _swapChainImages.size();
    _uniformBuffers.resize(n);
    _uniformBuffersMemory.resize(n);
    for (int i = 0; i < n; (i) += (1)) {
      auto [buf, mem] =
          createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                       ((VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) |
                        (VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)));
      _uniformBuffers[i] = buf;
      _uniformBuffersMemory[i] = mem;
    };
  }
  void createDescriptorSetLayout() {
    VkDescriptorSetLayoutBinding uboLayoutBinding = {};
    uboLayoutBinding.binding = 0;
    uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboLayoutBinding.descriptorCount = 1;
    uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    uboLayoutBinding.pImmutableSamplers = nullptr;
    {
      VkDescriptorSetLayoutCreateInfo info = {};
      info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
      info.bindingCount = 1;
      info.pBindings = &uboLayoutBinding;
      if (!((VK_SUCCESS) ==
            (vkCreateDescriptorSetLayout(_device, &info, nullptr,
                                         &_descriptorSetLayout)))) {
        throw std::runtime_error(
            "failed to (vkCreateDescriptorSetLayout _device &info nullptr      "
            "      &_descriptorSetLayout)");
      };
    };
  }
  void recreateSwapChain() {
    int width = 0;
    int height = 0;
    while ((((0) == (width)) || ((0) == (height)))) {
      glfwGetFramebufferSize(_window, &width, &height);
      glfwWaitEvents();
    };
    vkDeviceWaitIdle(_device);
    createSwapChain();
    createImageViews();
    createRenderPass();
    createGraphicsPipeline();
    createFramebuffers();
    createUniformBuffers();
    createDescriptorPool();
    createDescriptorSets();
    createCommandBuffers();
  }
  // shader stuff
  void createSyncObjects() {
    _imageAvailableSemaphores.resize(_MAX_FRAMES_IN_FLIGHT);
    _renderFinishedSemaphores.resize(_MAX_FRAMES_IN_FLIGHT);
    _inFlightFences.resize(_MAX_FRAMES_IN_FLIGHT);
    VkSemaphoreCreateInfo semaphoreInfo = {};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    VkFenceCreateInfo fenceInfo = {};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    for (int i = 0; i < _MAX_FRAMES_IN_FLIGHT; (i) += (1)) {
      if (!((VK_SUCCESS) ==
            (vkCreateSemaphore(_device, &semaphoreInfo, nullptr,
                               &(_imageAvailableSemaphores[i]))))) {
        throw std::runtime_error(
            "failed to (vkCreateSemaphore _device &semaphoreInfo nullptr       "
            "     (ref (aref _imageAvailableSemaphores i)))");
      };
      if (!((VK_SUCCESS) ==
            (vkCreateSemaphore(_device, &semaphoreInfo, nullptr,
                               &(_renderFinishedSemaphores[i]))))) {
        throw std::runtime_error(
            "failed to (vkCreateSemaphore _device &semaphoreInfo nullptr       "
            "     (ref (aref _renderFinishedSemaphores i)))");
      };
      if (!((VK_SUCCESS) == (vkCreateFence(_device, &fenceInfo, nullptr,
                                           &(_inFlightFences[i]))))) {
        throw std::runtime_error(
            "failed to (vkCreateFence _device &fenceInfo nullptr            "
            "(ref (aref _inFlightFences i)))");
      };
    }
  }
  void createCommandBuffers() {
    _commandBuffers.resize(_swapChainFramebuffers.size());
    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = _commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = _commandBuffers.size();
    if (!((VK_SUCCESS) == (vkAllocateCommandBuffers(_device, &allocInfo,
                                                    _commandBuffers.data())))) {
      throw std::runtime_error("failed to (vkAllocateCommandBuffers _device "
                               "&allocInfo (_commandBuffers.data))");
    };
    for (int i = 0; i < _commandBuffers.size(); (i) += (1)) {
      VkCommandBufferBeginInfo beginInfo = {};
      beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
      beginInfo.flags = 0;
      beginInfo.pInheritanceInfo = nullptr;
      if (!((VK_SUCCESS) ==
            (vkBeginCommandBuffer(_commandBuffers[i], &beginInfo)))) {
        throw std::runtime_error("failed to (vkBeginCommandBuffer (aref "
                                 "_commandBuffers i) &beginInfo)");
      };
      VkClearValue clearColor = {(0.0e+0f), (0.0e+0f), (0.0e+0f), (1.e+0f)};
      VkRenderPassBeginInfo renderPassInfo = {};
      renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
      renderPassInfo.renderPass = _renderPass;
      renderPassInfo.framebuffer = _swapChainFramebuffers[i];
      renderPassInfo.renderArea.offset = {0, 0};
      renderPassInfo.renderArea.extent = _swapChainExtent;
      renderPassInfo.clearValueCount = 1;
      renderPassInfo.pClearValues = &clearColor;
      vkCmdBeginRenderPass(_commandBuffers[i], &renderPassInfo,
                           VK_SUBPASS_CONTENTS_INLINE);
      vkCmdBindPipeline(_commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS,
                        _graphicsPipeline);
      VkBuffer vertexBuffers[] = {_vertexBuffer};
      VkDeviceSize offsets[] = {0};
      vkCmdBindVertexBuffers(_commandBuffers[i], 0, 1, vertexBuffers, offsets);
      vkCmdBindIndexBuffer(_commandBuffers[i], _indexBuffer, 0,
                           VK_INDEX_TYPE_UINT16);
      vkCmdBindDescriptorSets(_commandBuffers[i],
                              VK_PIPELINE_BIND_POINT_GRAPHICS, _pipelineLayout,
                              0, 1, &(_descriptorSets[i]), 0, nullptr);
      vkCmdDrawIndexed(_commandBuffers[i],
                       static_cast<uint32_t>(g_indices.size()), 1, 0, 0, 0);
      vkCmdEndRenderPass(_commandBuffers[i]);
      if (!((VK_SUCCESS) == (vkEndCommandBuffer(_commandBuffers[i])))) {
        throw std::runtime_error(
            "failed to (vkEndCommandBuffer (aref _commandBuffers i))");
      };
    }
  }
  void createCommandPool() {
    auto queueFamilyIndices = findQueueFamilies(_physicalDevice, _surface);
    VkCommandPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();
    poolInfo.flags = 0;
    if (!((VK_SUCCESS) ==
          (vkCreateCommandPool(_device, &poolInfo, nullptr, &_commandPool)))) {
      throw std::runtime_error("failed to (vkCreateCommandPool _device "
                               "&poolInfo nullptr &_commandPool)");
    };
  }
  void createFramebuffers() {
    _swapChainFramebuffers.resize(_swapChainImageViews.size());
    for (int i = 0; i < _swapChainImageViews.size(); (i) += (1)) {
      VkImageView attachments[] = {_swapChainImageViews[i]};
      VkFramebufferCreateInfo framebufferInfo = {};
      framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
      framebufferInfo.renderPass = _renderPass;
      framebufferInfo.attachmentCount = 1;
      framebufferInfo.pAttachments = attachments;
      framebufferInfo.width = _swapChainExtent.width;
      framebufferInfo.height = _swapChainExtent.height;
      framebufferInfo.layers = 1;
      if (!((VK_SUCCESS) ==
            (vkCreateFramebuffer(_device, &framebufferInfo, nullptr,
                                 &(_swapChainFramebuffers[i]))))) {
        throw std::runtime_error(
            "failed to (vkCreateFramebuffer _device &framebufferInfo nullptr   "
            "         (ref (aref _swapChainFramebuffers i)))");
      };
    }
  }
  void createRenderPass() {
    VkAttachmentDescription colorAttachment = {};
    colorAttachment.format = _swapChainImageFormat;
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    VkAttachmentReference colorAttachmentRef = {};
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    VkSubpassDescription subpass = {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentRef;
    VkSubpassDependency dependency = {};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.srcAccessMask = 0;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.dstAccessMask = ((VK_ACCESS_COLOR_ATTACHMENT_READ_BIT) |
                                (VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT));
    VkRenderPassCreateInfo renderPassInfo = {};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = 1;
    renderPassInfo.pAttachments = &colorAttachment;
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    renderPassInfo.dependencyCount = 1;
    renderPassInfo.pDependencies = &dependency;
    if (!((VK_SUCCESS) == (vkCreateRenderPass(_device, &renderPassInfo, nullptr,
                                              &_renderPass)))) {
      throw std::runtime_error("failed to (vkCreateRenderPass _device "
                               "&renderPassInfo nullptr &_renderPass)");
    };
  }
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
    auto bindingDescription = Vertex::getBindingDescription();
    auto attributeDescriptions = Vertex::getAttributeDescriptions();
    VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
    vertexInputInfo.sType =
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
    vertexInputInfo.vertexAttributeDescriptionCount =
        static_cast<uint32_t>(attributeDescriptions.size());
    vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();
    VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
    inputAssembly.sType =
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    inputAssembly.primitiveRestartEnable = VK_FALSE;
    VkViewport viewport = {};
    viewport.x = (0.0e+0f);
    viewport.y = (0.0e+0f);
    viewport.width = (((1.e+0f)) * (_swapChainExtent.width));
    viewport.height = (((1.e+0f)) * (_swapChainExtent.height));
    viewport.minDepth = (0.0e+0f);
    viewport.maxDepth = (1.e+0f);
    VkRect2D scissor = {};
    scissor.offset = {0, 0};
    scissor.extent = _swapChainExtent;
    VkPipelineViewportStateCreateInfo viewPortState = {};
    viewPortState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewPortState.viewportCount = 1;
    viewPortState.pViewports = &viewport;
    viewPortState.scissorCount = 1;
    viewPortState.pScissors = &scissor;
    VkPipelineRasterizationStateCreateInfo rasterizer = {};
    rasterizer.sType =
        VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = (1.e+0f);
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
    rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
    rasterizer.depthBiasEnable = VK_FALSE;
    rasterizer.depthBiasConstantFactor = (0.0e+0f);
    rasterizer.depthBiasClamp = (0.0e+0f);
    rasterizer.depthBiasSlopeFactor = (0.0e+0f);
    VkPipelineMultisampleStateCreateInfo multisampling = {};
    multisampling.sType =
        VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    multisampling.minSampleShading = (1.e+0f);
    multisampling.pSampleMask = nullptr;
    multisampling.alphaToCoverageEnable = VK_FALSE;
    multisampling.alphaToOneEnable = VK_FALSE;
    VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
    colorBlendAttachment.colorWriteMask =
        ((VK_COLOR_COMPONENT_R_BIT) | (VK_COLOR_COMPONENT_G_BIT) |
         (VK_COLOR_COMPONENT_B_BIT) | (VK_COLOR_COMPONENT_A_BIT));
    colorBlendAttachment.blendEnable = VK_FALSE;
    colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
    colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
    colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
    colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;
    VkPipelineColorBlendStateCreateInfo colorBlending = {};
    colorBlending.sType =
        VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.logicOp = VK_LOGIC_OP_COPY;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;
    colorBlending.blendConstants[0] = (0.0e+0f);
    colorBlending.blendConstants[1] = (0.0e+0f);
    colorBlending.blendConstants[2] = (0.0e+0f);
    colorBlending.blendConstants[3] = (0.0e+0f);
    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &_descriptorSetLayout;
    pipelineLayoutInfo.pushConstantRangeCount = 0;
    pipelineLayoutInfo.pPushConstantRanges = nullptr;
    if (!((VK_SUCCESS) ==
          (vkCreatePipelineLayout(_device, &pipelineLayoutInfo, nullptr,
                                  &_pipelineLayout)))) {
      throw std::runtime_error(
          "failed to (vkCreatePipelineLayout _device &pipelineLayoutInfo "
          "nullptr            &_pipelineLayout)");
    };
    VkGraphicsPipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = shaderStages;
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewPortState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = nullptr;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.pDynamicState = nullptr;
    pipelineInfo.layout = _pipelineLayout;
    pipelineInfo.renderPass = _renderPass;
    pipelineInfo.subpass = 0;
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
    pipelineInfo.basePipelineIndex = -1;
    if (!((VK_SUCCESS) ==
          (vkCreateGraphicsPipelines(_device, VK_NULL_HANDLE, 1, &pipelineInfo,
                                     nullptr, &_graphicsPipeline)))) {
      throw std::runtime_error(
          "failed to (vkCreateGraphicsPipelines _device VK_NULL_HANDLE 1 "
          "&pipelineInfo            nullptr &_graphicsPipeline)");
    };
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
      throw std::runtime_error("failed to (vkCreateShaderModule _device "
                               "&createInfo nullptr &shaderModule)");
    };
    return shaderModule;
  };
  void createSurface() {
    // initialize _surface member
    // must be destroyed before the instance is destroyed
    if (!((VK_SUCCESS) ==
          (glfwCreateWindowSurface(_instance, _window, nullptr, &_surface)))) {
      throw std::runtime_error("failed to (glfwCreateWindowSurface _instance "
                               "_window nullptr &_surface)");
    };
  }
  void createSwapChain() {
    auto swapChainSupport = querySwapChainSupport(_physicalDevice, _surface);
    auto surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
    auto presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
    auto extent = chooseSwapExtent(swapChainSupport.capabilities, _window);
    auto imageCount = ((swapChainSupport.capabilities.minImageCount) + (1));
    auto indices = findQueueFamilies(_physicalDevice, _surface);
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
      throw std::runtime_error("failed to (vkCreateSwapchainKHR _device "
                               "&createInfo nullptr &_swapChain)");
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
        throw std::runtime_error(
            "failed to (vkCreateImageView _device &createInfo nullptr          "
            "  (ref (aref _swapChainImageViews i)))");
      };
    }
  };
  void createLogicalDevice() {
    // initialize members _device and _graphicsQueue
    auto indices = findQueueFamilies(_physicalDevice, _surface);
    float queuePriority = (1.e+0f);
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
      throw std::runtime_error("failed to (vkCreateDevice _physicalDevice "
                               "&createInfo nullptr &_device)");
    };
    vkGetDeviceQueue(_device, indices.graphicsFamily.value(), 0,
                     &_graphicsQueue);
    vkGetDeviceQueue(_device, indices.presentFamily.value(), 0, &_presentQueue);
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
      if (isDeviceSuitable(device, _surface, _deviceExtensions)) {
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
      drawFrame();
    }
    vkDeviceWaitIdle(_device);
  }
  void drawFrame() {
    vkWaitForFences(_device, 1, &(_inFlightFences[_currentFrame]), VK_TRUE,
                    UINT64_MAX);
    uint32_t imageIndex = 0;
    auto result = vkAcquireNextImageKHR(
        _device, _swapChain, UINT64_MAX,
        _imageAvailableSemaphores[_currentFrame], VK_NULL_HANDLE, &imageIndex);
    if ((VK_ERROR_OUT_OF_DATE_KHR) == (result)) {
      recreateSwapChain();
      return;
    };
    if (!((((VK_SUCCESS) == (result)) || ((VK_SUBOPTIMAL_KHR) == (result))))) {
      throw std::runtime_error("failed to acquire swap chain image.");
    };
    VkSemaphore waitSemaphores[] = {_imageAvailableSemaphores[_currentFrame]};
    VkSemaphore signalSemaphores[] = {_renderFinishedSemaphores[_currentFrame]};
    VkPipelineStageFlags waitStages[] = {
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    updateUniformBuffer(imageIndex);
    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = waitSemaphores;
    submitInfo.pWaitDstStageMask = waitStages;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &(_commandBuffers[imageIndex]);
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = signalSemaphores;
    vkResetFences(_device, 1, &(_inFlightFences[_currentFrame]));
    if (!((VK_SUCCESS) == (vkQueueSubmit(_graphicsQueue, 1, &submitInfo,
                                         _inFlightFences[_currentFrame])))) {
      throw std::runtime_error(
          "failed to (vkQueueSubmit _graphicsQueue 1 &submitInfo            "
          "(aref _inFlightFences _currentFrame))");
    };
    VkSwapchainKHR swapChains[] = {_swapChain};
    VkPresentInfoKHR presentInfo = {};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = signalSemaphores;
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = swapChains;
    presentInfo.pImageIndices = &imageIndex;
    presentInfo.pResults = nullptr;
    {
      auto result2 = vkQueuePresentKHR(_presentQueue, &presentInfo);
      if ((((VK_SUBOPTIMAL_KHR) == (result2)) ||
           ((VK_ERROR_OUT_OF_DATE_KHR) == (result2)) ||
           (_framebufferResized))) {
        _framebufferResized = false;
        recreateSwapChain();
      } else {
        if (!((VK_SUCCESS) == (result2))) {
          throw std::runtime_error("fialed to present swap chain image.");
        };
      };
    };
    _currentFrame = ((1) + (_currentFrame)) % _MAX_FRAMES_IN_FLIGHT;
  }
  void updateUniformBuffer(uint32_t currentImage) {
    static auto startTime = std::chrono::high_resolution_clock::now();
    auto currentTime = std::chrono::high_resolution_clock::now();
    float time = std::chrono::duration<float, std::chrono::seconds::period>(
                     ((currentTime) - (startTime)))
                     .count();
    const auto zAxis = glm::vec3((0.0e+0f), (0.0e+0f), (1.e+0f));
    const auto identityMatrix = glm::mat4((1.e+0f));
    const auto angularRate = glm::radians((9.e+1f));
    auto rotationAngle = ((time) * (angularRate));
    UniformBufferObject ubo = {};
    ubo.model = glm::rotate(identityMatrix, rotationAngle, zAxis);
    ubo.view = glm::perspective(
        glm::radians((4.5e+1f)),
        ((_swapChainExtent.width) / ((((1.e+0f)) * (_swapChainExtent.height)))),
        (1.e-1f), (1.e+1f));
    ubo.proj[1][1] = (-(ubo.proj[1][1]));
    void *data = 0;
    vkMapMemory(_device, _uniformBuffersMemory[currentImage], 0, sizeof(ubo), 0,
                &data);
    memcpy(data, &ubo, sizeof(ubo));
    vkUnmapMemory(_device, _uniformBuffersMemory[currentImage]);
  }
  void cleanupSwapChain() {
    for (auto &b : _swapChainFramebuffers) {
      vkDestroyFramebuffer(_device, b, nullptr);
    };
    vkFreeCommandBuffers(_device, _commandPool,
                         static_cast<uint32_t>(_commandBuffers.size()),
                         _commandBuffers.data());
    vkDestroyPipeline(_device, _graphicsPipeline, nullptr);
    vkDestroyPipelineLayout(_device, _pipelineLayout, nullptr);
    vkDestroyRenderPass(_device, _renderPass, nullptr);
    for (auto &view : _swapChainImageViews) {
      vkDestroyImageView(_device, view, nullptr);
    };
    vkDestroySwapchainKHR(_device, _swapChain, nullptr);
    for (int i = 0; i < _swapChainImages.size(); (i) += (1)) {
      vkDestroyBuffer(_device, _uniformBuffers[i], nullptr);
      vkFreeMemory(_device, _uniformBuffersMemory[i], nullptr);
    }
    vkDestroyDescriptorPool(_device, _descriptorPool, nullptr);
  }
  void cleanup() {
    cleanupSwapChain();
    vkDestroyDescriptorSetLayout(_device, _descriptorSetLayout, nullptr);
    vkDestroyBuffer(_device, _vertexBuffer, nullptr);
    vkFreeMemory(_device, _vertexBufferMemory, nullptr);
    vkDestroyBuffer(_device, _indexBuffer, nullptr);
    vkFreeMemory(_device, _indexBufferMemory, nullptr);
    for (int i = 0; i < _MAX_FRAMES_IN_FLIGHT; (i) += (1)) {
      vkDestroySemaphore(_device, _renderFinishedSemaphores[i], nullptr);
      vkDestroySemaphore(_device, _imageAvailableSemaphores[i], nullptr);
      vkDestroyFence(_device, _inFlightFences[i], nullptr);
    }
    vkDestroyCommandPool(_device, _commandPool, nullptr);
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