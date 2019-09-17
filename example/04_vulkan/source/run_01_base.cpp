// https://vulkan-tutorial.com/en/Drawing_a_triangle/Setup/Base_code
// https://vulkan-tutorial.com/en/Drawing_a_triangle/Setup/Validation_layers
// https://gpuopen.com/understanding-vulkan-objects/
/* g++ -std=c++17 run_01_base.cpp  `pkg-config --static --libs glfw3` -lvulkan
 * -o run_01_base -pedantic -Wall -Wextra -Wcast-align -Wcast-qual
 * -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2 -Winit-self
 * -Wlogical-op -Wmissing-declarations -Wmissing-include-dirs -Wnoexcept
 * -Wold-style-cast -Woverloaded-virtual -Wredundant-decls -Wshadow
 * -Wsign-conversion -Wsign-promo -Wstrict-null-sentinel -Wstrict-overflow=5
 * -Wswitch-default -Wundef -march=native -O2 -g  -ftime-report */

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
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
;
#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"
#include <unordered_map>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>
;
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
  glm::mat4 model;
  glm::mat4 view;
  glm::mat4 proj;
};
typedef struct UniformBufferObject UniformBufferObject;
struct Vertex {
  glm::vec3 pos;
  glm::vec3 color;
  glm::vec2 texCoord;
  static VkVertexInputBindingDescription getBindingDescription();
  static std::array<VkVertexInputAttributeDescription, 3>
  getAttributeDescriptions();
  bool operator==(const Vertex &other) const;
};
typedef struct Vertex Vertex;
bool Vertex::operator==(const Vertex &other) const {
  return (((pos) == (other.pos)) && ((color) == (other.color)) &&
          ((texCoord) == (other.texCoord)));
};
template <> struct std::hash<Vertex> {
  size_t operator()(Vertex const &vertex) const {
    return ((std::hash<glm::vec3>()(vertex.pos)) ^
            (((std::hash<glm::vec3>()(vertex.color)) << (1)) >> (1)) ^
            ((std::hash<glm::vec2>()(vertex.texCoord)) << (1)));
  };
};
VkVertexInputBindingDescription Vertex::getBindingDescription() {
  VkVertexInputBindingDescription bindingDescription = {};
  bindingDescription.binding = 0;
  bindingDescription.stride = sizeof(Vertex);
  bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
  return bindingDescription;
}
std::vector<Vertex> g_vertices;
std::vector<uint32_t> g_indices;
std::array<VkVertexInputAttributeDescription, 3>
Vertex::getAttributeDescriptions() {
  std::array<VkVertexInputAttributeDescription, 3> attributeDescriptions = {};
  attributeDescriptions[0].binding = 0;
  attributeDescriptions[0].location = 0;
  attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
  attributeDescriptions[0].offset = offsetof(Vertex, pos);
  attributeDescriptions[1].binding = 0;
  attributeDescriptions[1].location = 1;
  attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
  attributeDescriptions[1].offset = offsetof(Vertex, color);
  attributeDescriptions[2].binding = 0;
  attributeDescriptions[2].location = 2;
  attributeDescriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
  attributeDescriptions[2].offset = offsetof(Vertex, texCoord);
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
    VkExtent2D actualExtent = {static_cast<uint32_t>(width),
                               static_cast<uint32_t>(height)};
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
  VkPhysicalDeviceFeatures supportedFeatures;
  vkGetPhysicalDeviceFeatures(device, &supportedFeatures);
  return ((indices.isComplete()) && (supportedFeatures.samplerAnisotropy) &&
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
  VkSampleCountFlagBits _msaaSamples;
  VkImage _colorImage;
  VkDeviceMemory _colorImageMemory;
  VkImageView _colorImageView;
  uint32_t _mipLevels;
  VkImage _textureImage;
  VkDeviceMemory _textureImageMemory;
  VkImageView _textureImageView;
  VkSampler _textureSampler;
  VkImage _depthImage;
  VkDeviceMemory _depthImageMemory;
  VkImageView _depthImageView;
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
    {
      VkInstanceCreateInfo info = {};
      info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
      info.pApplicationInfo = &appInfo;
      info.enabledExtensionCount = glfwExtensionCount;
      info.ppEnabledExtensionNames = glfwExtensions;
      info.enabledLayerCount = static_cast<uint32_t>(_validationLayers.size());
      info.ppEnabledLayerNames = _validationLayers.data();
      if (!((VK_SUCCESS) == (vkCreateInstance(&info, nullptr, &_instance)))) {
        throw std::runtime_error(
            "failed to (vkCreateInstance &info nullptr &_instance)");
      };
      (std::cout) << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                  << (__func__) << (" create instance _instance=")
                  << (_instance) << (std::endl);
    };
  }
  std::tuple<VkBuffer, VkDeviceMemory>
  createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
               VkMemoryPropertyFlags properties) {
    VkBuffer buffer;
    VkDeviceMemory bufferMemory;
    {
      VkBufferCreateInfo info = {};
      info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
      info.size = size;
      info.usage = usage;
      info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
      info.flags = 0;
      if (!((VK_SUCCESS) ==
            (vkCreateBuffer(_device, &info, nullptr, &buffer)))) {
        throw std::runtime_error(
            "failed to (vkCreateBuffer _device &info nullptr &buffer)");
      };
      (std::cout) << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                  << (__func__) << (" create buffer buffer=") << (buffer)
                  << (std::endl);
    };
    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(_device, buffer, &memReq);
    {
      VkMemoryAllocateInfo info = {};
      info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
      info.allocationSize = memReq.size;
      info.memoryTypeIndex = findMemoryType(memReq.memoryTypeBits, properties);
      if (!((VK_SUCCESS) ==
            (vkAllocateMemory(_device, &info, nullptr, &bufferMemory)))) {
        throw std::runtime_error(
            "failed to (vkAllocateMemory _device &info nullptr &bufferMemory)");
      };
      (std::cout) << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                  << (__func__) << (" allocate memory bufferMemory=")
                  << (bufferMemory) << (std::endl);
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
  VkCommandBuffer beginSingleTimeCommands() {
    VkCommandBuffer commandBuffer;
    {
      VkCommandBufferAllocateInfo info = {};
      info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
      info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
      info.commandPool = _commandPool;
      info.commandBufferCount = 1;
      vkAllocateCommandBuffers(_device, &info, &commandBuffer);
      (std::cout) << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                  << (__func__) << (" allocate command-buffer") << (std::endl);
    };
    {
      VkCommandBufferBeginInfo info = {};
      info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
      info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
      vkBeginCommandBuffer(commandBuffer, &info);
      (std::cout) << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                  << (__func__) << (" begin command-buffer commandBuffer=")
                  << (commandBuffer) << (std::endl);
    };
    return commandBuffer;
  }
  void endSingleTimeCommands(VkCommandBuffer commandBuffer) {
    vkEndCommandBuffer(commandBuffer);
    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    vkQueueSubmit(_graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(_graphicsQueue);
    vkFreeCommandBuffers(_device, _commandPool, 1, &commandBuffer);
    (std::cout) << ("endSingleTimeCommands ") << (commandBuffer) << (std::endl);
  }
  void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
    auto commandBuffer = beginSingleTimeCommands();
    VkBufferCopy copyRegion = {};
    copyRegion.srcOffset = 0;
    copyRegion.dstOffset = 0;
    copyRegion.size = size;
    vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);
    endSingleTimeCommands(commandBuffer);
  }
  uint32_t findMemoryType(uint32_t typeFilter,
                          VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties ps;
    vkGetPhysicalDeviceMemoryProperties(_physicalDevice, &ps);
    for (int i = 0; i < ps.memoryTypeCount; (i) += (1)) {
      if ((((((1) << (i)) & (typeFilter))) &&
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
    (std::cout) << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" call createSwapChain") << (std::endl);
    createSwapChain();
    (std::cout) << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" call createImageViews") << (std::endl);
    createImageViews();
    (std::cout) << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" call createRenderPass") << (std::endl);
    createRenderPass();
    (std::cout) << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" call createDescriptorSetLayout") << (std::endl);
    createDescriptorSetLayout();
    (std::cout) << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" call createGraphicsPipeline") << (std::endl);
    createGraphicsPipeline();
    (std::cout) << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" call createCommandPool") << (std::endl);
    createCommandPool();
    (std::cout) << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" call createColorResources") << (std::endl);
    createColorResources();
    (std::cout) << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" call createDepthResources") << (std::endl);
    createDepthResources();
    (std::cout) << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" call createFramebuffers") << (std::endl);
    createFramebuffers();
    (std::cout) << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" call createTextureImage") << (std::endl);
    createTextureImage();
    (std::cout) << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" call createTextureImageView") << (std::endl);
    createTextureImageView();
    (std::cout) << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" call createTextureSampler") << (std::endl);
    createTextureSampler();
    (std::cout) << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" call loadModel") << (std::endl);
    loadModel();
    (std::cout) << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" call createVertexBuffer") << (std::endl);
    createVertexBuffer();
    (std::cout) << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" call createIndexBuffer") << (std::endl);
    createIndexBuffer();
    (std::cout) << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" call createUniformBuffers") << (std::endl);
    createUniformBuffers();
    (std::cout) << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" call createDescriptorPool") << (std::endl);
    createDescriptorPool();
    (std::cout) << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" call createDescriptorSets") << (std::endl);
    createDescriptorSets();
    (std::cout) << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" call createCommandBuffers") << (std::endl);
    createCommandBuffers();
    (std::cout) << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" call createSyncObjects") << (std::endl);
    createSyncObjects();
  }
  void loadModel() {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warning;
    std::string err;
    if (!(tinyobj::LoadObj(&attrib, &shapes, &materials, &warning, &err,
                           "chalet.obj"))) {
      throw std::runtime_error(((warning) + (err)));
    };
    std::unordered_map<Vertex, uint32_t> uniqueVertices = {};
    for (auto &shape : shapes) {
      for (auto &index : shape.mesh.indices) {
        Vertex vertex = {};
        vertex.pos = {attrib.vertices[((0) + (((3) * (index.vertex_index))))],
                      attrib.vertices[((1) + (((3) * (index.vertex_index))))],
                      attrib.vertices[((2) + (((3) * (index.vertex_index))))]};
        vertex.texCoord = {
            attrib.texcoords[((0) + (((2) * (index.texcoord_index))))],
            (((1.e+0f)) -
             (attrib.texcoords[((1) + (((2) * (index.texcoord_index))))]))};
        vertex.color = {(1.e+0f), (1.e+0f), (1.e+0f)};
        if ((0) == (uniqueVertices.count(vertex))) {
          uniqueVertices[vertex] = static_cast<uint32_t>(g_vertices.size());
          g_vertices.push_back(vertex);
        };
        g_indices.push_back(uniqueVertices[vertex]);
      };
    };
  };
  VkFormat findSupportedFormat(const std::vector<VkFormat> &candidates,
                               VkImageTiling tiling,
                               VkFormatFeatureFlags features) {
    for (auto &format : candidates) {
      VkFormatProperties props;
      vkGetPhysicalDeviceFormatProperties(_physicalDevice, format, &props);
      if ((((VK_IMAGE_TILING_LINEAR) == (tiling)) &&
           ((features) == (((features) & (props.linearTilingFeatures)))))) {
        return format;
      };
      if ((((VK_IMAGE_TILING_OPTIMAL) == (tiling)) &&
           ((features) == (((features) & (props.optimalTilingFeatures)))))) {
        return format;
      };
    };
    throw std::runtime_error("failed to find supported format!");
  }
  VkFormat findDepthFormat() {
    return findSupportedFormat({VK_FORMAT_D32_SFLOAT,
                                VK_FORMAT_D32_SFLOAT_S8_UINT,
                                VK_FORMAT_D24_UNORM_S8_UINT},
                               VK_IMAGE_TILING_OPTIMAL,
                               VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
  }
  bool hasStencilComponent(VkFormat format) {
    return (((VK_FORMAT_D32_SFLOAT_S8_UINT) == (format)) ||
            ((VK_FORMAT_D24_UNORM_S8_UINT) == (format)));
  }
  void createDepthResources() {
    auto depthFormat = findDepthFormat();
    auto [depthImage, depthImageMemory] =
        createImage(_swapChainExtent.width, _swapChainExtent.height, 1,
                    VK_SAMPLE_COUNT_1_BIT, depthFormat, VK_IMAGE_TILING_OPTIMAL,
                    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    _depthImage = depthImage;
    _depthImageMemory = depthImageMemory;
    _depthImageView =
        createImageView(_depthImage, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT, 1);
    transitionImageLayout(_depthImage, depthFormat, VK_IMAGE_LAYOUT_UNDEFINED,
                          VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, 1);
  };
  void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width,
                         uint32_t height) {
    auto commandBuffer = beginSingleTimeCommands();
    VkBufferImageCopy region = {};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = {0, 0, 0};
    region.imageExtent = {width, height, 1};
    vkCmdCopyBufferToImage(commandBuffer, buffer, image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    endSingleTimeCommands(commandBuffer);
  }
  void transitionImageLayout(VkImage image, VkFormat format,
                             VkImageLayout oldLayout, VkImageLayout newLayout,
                             uint32_t mipLevels) {
    auto commandBuffer = beginSingleTimeCommands();
    VkImageMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = mipLevels;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = 0;
    if ((VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) == (newLayout)) {
      barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
      if (hasStencilComponent(format)) {
        barrier.subresourceRange.aspectMask =
            ((barrier.subresourceRange.aspectMask) |
             (VK_IMAGE_ASPECT_STENCIL_BIT));
      };
    } else {
      barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    }
    VkPipelineStageFlags srcStage;
    VkPipelineStageFlags dstStage;
    if ((((VK_IMAGE_LAYOUT_UNDEFINED) == (oldLayout)) &&
         ((VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) == (newLayout)))) {
      barrier.srcAccessMask = 0;
      barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
      dstStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    } else {
      if ((((VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) == (oldLayout)) &&
           ((VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) == (newLayout)))) {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        srcStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        dstStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
      } else {
        if ((((VK_IMAGE_LAYOUT_UNDEFINED) == (oldLayout)) &&
             ((VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) ==
              (newLayout)))) {
          barrier.srcAccessMask = 0;
          barrier.dstAccessMask =
              ((VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT) |
               (VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT));
          srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
          dstStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        } else {
          if ((((VK_IMAGE_LAYOUT_UNDEFINED) == (oldLayout)) &&
               ((VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) == (newLayout)))) {
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = ((VK_ACCESS_COLOR_ATTACHMENT_READ_BIT) |
                                     (VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT));
            srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            dstStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
          } else {
            throw std::invalid_argument("unsupported layout transition.");
          }
        }
      }
    };
    vkCmdPipelineBarrier(commandBuffer, srcStage, dstStage, 0, 0, nullptr, 0,
                         nullptr, 1, &barrier);
    endSingleTimeCommands(commandBuffer);
  }
  std::tuple<VkImage, VkDeviceMemory>
  createImage(uint32_t width, uint32_t height, uint32_t mipLevels,
              VkSampleCountFlagBits numSamples, VkFormat format,
              VkImageTiling tiling, VkImageUsageFlags usage,
              VkMemoryPropertyFlags properties) {
    VkImage image;
    VkDeviceMemory imageMemory;
    {
      VkImageCreateInfo info = {};
      info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
      info.imageType = VK_IMAGE_TYPE_2D;
      info.extent.width = width;
      info.extent.height = height;
      info.extent.depth = 1;
      info.mipLevels = mipLevels;
      info.arrayLayers = 1;
      info.format = format;
      info.tiling = tiling;
      info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      info.usage = usage;
      info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
      info.samples = numSamples;
      info.flags = 0;
      if (!((VK_SUCCESS) == (vkCreateImage(_device, &info, nullptr, &image)))) {
        throw std::runtime_error(
            "failed to (vkCreateImage _device &info nullptr &image)");
      };
      (std::cout) << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                  << (__func__) << (" create image image=") << (image)
                  << (std::endl);
    };
    VkMemoryRequirements memReq;
    vkGetImageMemoryRequirements(_device, image, &memReq);
    {
      VkMemoryAllocateInfo info = {};
      info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
      info.allocationSize = memReq.size;
      info.memoryTypeIndex = findMemoryType(memReq.memoryTypeBits, properties);
      if (!((VK_SUCCESS) ==
            (vkAllocateMemory(_device, &info, nullptr, &imageMemory)))) {
        throw std::runtime_error(
            "failed to (vkAllocateMemory _device &info nullptr &imageMemory)");
      };
      (std::cout) << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                  << (__func__) << (" allocate memory imageMemory=")
                  << (imageMemory) << (std::endl);
    };
    vkBindImageMemory(_device, image, imageMemory, 0);
    return std::make_tuple(image, imageMemory);
  }
  void createTextureImage() {
    // uses command buffers
    int texWidth = 0;
    int texHeight = 0;
    int texChannels = 0;
    auto pixels = stbi_load("chalet.jpg", &texWidth, &texHeight, &texChannels,
                            STBI_rgb_alpha);
    VkDeviceSize imageSize = ((texWidth) * (texHeight) * (4));
    if (!(pixels)) {
      throw std::runtime_error("failed to load texture image.");
    };
    _mipLevels = static_cast<uint32_t>(
        ((1) + (std::floor(std::log2(std::max(texWidth, texHeight))))));
    // width    mipLevels
    // 2        2
    // 4        3
    // 16       5
    // 32       6
    // 128      8
    // 255      8
    // 256      9
    // 257      9
    // 512      10
    // 1024     11      ;
    auto [stagingBuffer, stagingBufferMemory] =
        createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     ((VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) |
                      (VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)));
    void *data = nullptr;
    vkMapMemory(_device, stagingBufferMemory, 0, imageSize, 0, &data);
    memcpy(data, pixels, static_cast<size_t>(imageSize));
    vkUnmapMemory(_device, stagingBufferMemory);
    stbi_image_free(pixels);
    auto [image, imageMemory] = createImage(
        texWidth, texHeight, _mipLevels, VK_SAMPLE_COUNT_1_BIT,
        VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_TILING_OPTIMAL,
        ((VK_IMAGE_USAGE_TRANSFER_DST_BIT) | (VK_IMAGE_USAGE_TRANSFER_SRC_BIT) |
         (VK_IMAGE_USAGE_SAMPLED_BIT)),
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    _textureImage = image;
    _textureImageMemory = imageMemory;
    transitionImageLayout(_textureImage, VK_FORMAT_R8G8B8A8_UNORM,
                          VK_IMAGE_LAYOUT_UNDEFINED,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, _mipLevels);
    copyBufferToImage(stagingBuffer, _textureImage,
                      static_cast<uint32_t>(texWidth),
                      static_cast<uint32_t>(texHeight));
    vkDestroyBuffer(_device, stagingBuffer, nullptr);
    vkFreeMemory(_device, stagingBufferMemory, nullptr);
    generateMipmaps(_textureImage, VK_FORMAT_R8G8B8A8_UNORM, texWidth,
                    texHeight, _mipLevels);
  }
  void generateMipmaps(VkImage image, VkFormat imageFormat, int32_t texWidth,
                       int32_t texHeight, int32_t mipLevels) {
    (std::cout) << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" generateMipmaps") << (std::endl);
    VkFormatProperties formatProperties;
    vkGetPhysicalDeviceFormatProperties(_physicalDevice, imageFormat,
                                        &formatProperties);
    if (!(((formatProperties.optimalTilingFeatures) &
           (VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT)))) {
      throw std::runtime_error(
          "texture image format does not support linear blitting!");
    };
    auto commandBuffer = beginSingleTimeCommands();
    VkImageMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.image = image;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.subresourceRange.levelCount = 1;
    auto mipWidth = texWidth;
    auto mipHeight = texHeight;
    for (int i = 1; i < mipLevels; (i)++) {
      barrier.subresourceRange.baseMipLevel = ((i) - (1));
      barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
      barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
      barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
      (std::cout) << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" vkCmdPipelineBarrier ") << (i) << (std::endl);
      vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                           nullptr, 1, &barrier);
      auto dstOffsetx = 1;
      auto dstOffsety = 1;
      if (1 < mipWidth) {
        dstOffsetx = ((mipWidth) / (2));
      };
      if (1 < mipHeight) {
        dstOffsety = ((mipHeight) / (2));
      };
      VkImageBlit blit = {};
      blit.srcOffsets[0] = {0, 0, 0};
      blit.srcOffsets[1] = {mipWidth, mipHeight, 1};
      blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      blit.srcSubresource.mipLevel = ((i) - (1));
      blit.srcSubresource.baseArrayLayer = 0;
      blit.srcSubresource.layerCount = 1;
      blit.dstOffsets[0] = {0, 0, 0};
      blit.dstOffsets[1] = {dstOffsetx, dstOffsety, 1};
      blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      blit.dstSubresource.mipLevel = i;
      blit.dstSubresource.baseArrayLayer = 0;
      blit.dstSubresource.layerCount = 1;
      (std::cout) << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" vkCmdBlitImage") << (i) << (std::endl);
      vkCmdBlitImage(commandBuffer, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                     image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit,
                     VK_FILTER_LINEAR);
      barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
      barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
      barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
      (std::cout) << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" vkCmdPipelineBarrier") << (i) << (std::endl);
      vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr,
                           0, nullptr, 1, &barrier);
      if (1 < mipWidth) {
        mipWidth = ((mipWidth) / (2));
      };
      if (1 < mipHeight) {
        mipHeight = ((mipHeight) / (2));
      };
    };
    barrier.subresourceRange.baseMipLevel = ((_mipLevels) - (1));
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr,
                         0, nullptr, 1, &barrier);
    endSingleTimeCommands(commandBuffer);
  }
  void createTextureSampler() {
    {
      VkSamplerCreateInfo info = {};
      info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
      info.magFilter = VK_FILTER_LINEAR;
      info.minFilter = VK_FILTER_LINEAR;
      info.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
      info.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
      info.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
      info.anisotropyEnable = VK_TRUE;
      info.maxAnisotropy = 16;
      info.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
      info.unnormalizedCoordinates = VK_FALSE;
      info.compareEnable = VK_FALSE;
      info.compareOp = VK_COMPARE_OP_ALWAYS;
      info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
      info.mipLodBias = (0.0e+0f);
      info.minLod = (0.0e+0f);
      info.maxLod = static_cast<float>(_mipLevels);
      if (!((VK_SUCCESS) ==
            (vkCreateSampler(_device, &info, nullptr, &_textureSampler)))) {
        throw std::runtime_error("failed to (vkCreateSampler _device &info "
                                 "nullptr &_textureSampler)");
      };
      (std::cout) << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                  << (__func__) << (" create sampler _textureSampler=")
                  << (_textureSampler) << (std::endl);
    };
  }
  void createTextureImageView() {
    _textureImageView = createImageView(_textureImage, VK_FORMAT_R8G8B8A8_UNORM,
                                        VK_IMAGE_ASPECT_COLOR_BIT, _mipLevels);
  };
  void createDescriptorSets() {
    auto n = static_cast<uint32_t>(_swapChainImages.size());
    std::vector<VkDescriptorSetLayout> layouts(n, _descriptorSetLayout);
    _descriptorSets.resize(n);
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
      (std::cout) << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                  << (__func__) << (" allocate descriptor-set") << (std::endl);
    };
    for (int i = 0; i < n; (i) += (1)) {
      VkDescriptorBufferInfo bufferInfo = {};
      bufferInfo.buffer = _uniformBuffers[i];
      bufferInfo.offset = 0;
      bufferInfo.range = sizeof(UniformBufferObject);
      VkWriteDescriptorSet uboDescriptorWrite = {};
      uboDescriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      uboDescriptorWrite.dstSet = _descriptorSets[i];
      uboDescriptorWrite.dstBinding = 0;
      uboDescriptorWrite.dstArrayElement = 0;
      uboDescriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
      uboDescriptorWrite.descriptorCount = 1;
      uboDescriptorWrite.pBufferInfo = &bufferInfo;
      uboDescriptorWrite.pImageInfo = nullptr;
      uboDescriptorWrite.pTexelBufferView = nullptr;
      VkDescriptorImageInfo imageInfo = {};
      imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      imageInfo.imageView = _textureImageView;
      imageInfo.sampler = _textureSampler;
      VkWriteDescriptorSet samplerDescriptorWrite = {};
      samplerDescriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      samplerDescriptorWrite.dstSet = _descriptorSets[i];
      samplerDescriptorWrite.dstBinding = 1;
      samplerDescriptorWrite.dstArrayElement = 0;
      samplerDescriptorWrite.descriptorType =
          VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      samplerDescriptorWrite.descriptorCount = 1;
      samplerDescriptorWrite.pBufferInfo = nullptr;
      samplerDescriptorWrite.pImageInfo = &imageInfo;
      samplerDescriptorWrite.pTexelBufferView = nullptr;
      std::array<VkWriteDescriptorSet, 2> descriptorWrites = {
          uboDescriptorWrite, samplerDescriptorWrite};
      vkUpdateDescriptorSets(_device,
                             static_cast<uint32_t>(descriptorWrites.size()),
                             descriptorWrites.data(), 0, nullptr);
    };
  }
  void createDescriptorPool() {
    auto n = static_cast<uint32_t>(_swapChainImages.size());
    VkDescriptorPoolSize uboPoolSize = {};
    uboPoolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboPoolSize.descriptorCount = n;
    VkDescriptorPoolSize samplerPoolSize = {};
    samplerPoolSize.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    samplerPoolSize.descriptorCount = n;
    std::array<VkDescriptorPoolSize, 2> poolSizes = {uboPoolSize,
                                                     samplerPoolSize};
    {
      VkDescriptorPoolCreateInfo info = {};
      info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
      info.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
      info.pPoolSizes = poolSizes.data();
      info.maxSets = n;
      info.flags = 0;
      if (!((VK_SUCCESS) == (vkCreateDescriptorPool(_device, &info, nullptr,
                                                    &_descriptorPool)))) {
        throw std::runtime_error("failed to (vkCreateDescriptorPool _device "
                                 "&info nullptr &_descriptorPool)");
      };
      (std::cout) << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                  << (__func__) << (" create descriptor-pool _descriptorPool=")
                  << (_descriptorPool) << (std::endl);
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
    VkDescriptorSetLayoutBinding samplerLayoutBinding = {};
    samplerLayoutBinding.binding = 1;
    samplerLayoutBinding.descriptorCount = 1;
    samplerLayoutBinding.descriptorType =
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    samplerLayoutBinding.pImmutableSamplers = nullptr;
    samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    VkDescriptorSetLayoutBinding uboLayoutBinding = {};
    uboLayoutBinding.binding = 0;
    uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboLayoutBinding.descriptorCount = 1;
    uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    uboLayoutBinding.pImmutableSamplers = nullptr;
    std::array<VkDescriptorSetLayoutBinding, 2> bindings = {
        uboLayoutBinding, samplerLayoutBinding};
    {
      VkDescriptorSetLayoutCreateInfo info = {};
      info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
      info.bindingCount = static_cast<uint32_t>(bindings.size());
      info.pBindings = bindings.data();
      if (!((VK_SUCCESS) ==
            (vkCreateDescriptorSetLayout(_device, &info, nullptr,
                                         &_descriptorSetLayout)))) {
        throw std::runtime_error(
            "failed to (vkCreateDescriptorSetLayout _device &info nullptr      "
            "      &_descriptorSetLayout)");
      };
      (std::cout) << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                  << (__func__)
                  << (" create descriptor-set-layout _descriptorSetLayout=")
                  << (_descriptorSetLayout) << (std::endl);
    };
  }
  void recreateSwapChain() {
    (std::cout) << ("***** recreateSwapChain") << (std::endl);
    int width = 0;
    int height = 0;
    while ((((0) == (width)) || ((0) == (height)))) {
      glfwGetFramebufferSize(_window, &width, &height);
      glfwWaitEvents();
    };
    vkDeviceWaitIdle(_device);
    cleanupSwapChain();
    createSwapChain();
    createImageViews();
    createRenderPass();
    createGraphicsPipeline();
    createDepthResources();
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
    {
      VkCommandBufferAllocateInfo info = {};
      info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
      info.commandPool = _commandPool;
      info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
      info.commandBufferCount = _commandBuffers.size();
      if (!((VK_SUCCESS) == (vkAllocateCommandBuffers(
                                _device, &info, _commandBuffers.data())))) {
        throw std::runtime_error("failed to (vkAllocateCommandBuffers _device "
                                 "&info (_commandBuffers.data))");
      };
      (std::cout) << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                  << (__func__) << (" allocate command-buffer") << (std::endl);
    };
    for (int i = 0; i < _commandBuffers.size(); (i) += (1)) {
      {
        VkCommandBufferBeginInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        info.flags = 0;
        info.pInheritanceInfo = nullptr;
        if (!((VK_SUCCESS) ==
              (vkBeginCommandBuffer(_commandBuffers[i], &info)))) {
          throw std::runtime_error("failed to (vkBeginCommandBuffer (aref "
                                   "_commandBuffers i) &info)");
        };
        (std::cout) << (std::chrono::high_resolution_clock::now()
                            .time_since_epoch()
                            .count())
                    << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                    << (__func__)
                    << (" begin command-buffer (aref _commandBuffers i)=")
                    << (_commandBuffers[i]) << (std::endl);
      };
      VkClearValue clearColor = {};
      clearColor.color = {(0.0e+0f), (0.0e+0f), (0.0e+0f), (1.e+0f)};
      VkClearValue clearDepth = {};
      clearDepth.depthStencil = {(1.e+0f), 0};
      auto clearValues = std::array<VkClearValue, 2>({clearColor, clearDepth});
      VkRenderPassBeginInfo renderPassInfo = {};
      renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
      renderPassInfo.renderPass = _renderPass;
      renderPassInfo.framebuffer = _swapChainFramebuffers[i];
      renderPassInfo.renderArea.offset = {0, 0};
      renderPassInfo.renderArea.extent = _swapChainExtent;
      renderPassInfo.clearValueCount = clearValues.size();
      renderPassInfo.pClearValues = clearValues.data();
      vkCmdBeginRenderPass(_commandBuffers[i], &renderPassInfo,
                           VK_SUBPASS_CONTENTS_INLINE);
      vkCmdBindPipeline(_commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS,
                        _graphicsPipeline);
      VkBuffer vertexBuffers[] = {_vertexBuffer};
      VkDeviceSize offsets[] = {0};
      vkCmdBindVertexBuffers(_commandBuffers[i], 0, 1, vertexBuffers, offsets);
      vkCmdBindIndexBuffer(_commandBuffers[i], _indexBuffer, 0,
                           VK_INDEX_TYPE_UINT32);
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
    {
      VkCommandPoolCreateInfo info = {};
      info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
      info.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();
      info.flags = 0;
      if (!((VK_SUCCESS) ==
            (vkCreateCommandPool(_device, &info, nullptr, &_commandPool)))) {
        throw std::runtime_error("failed to (vkCreateCommandPool _device &info "
                                 "nullptr &_commandPool)");
      };
      (std::cout) << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                  << (__func__) << (" create command-pool _commandPool=")
                  << (_commandPool) << (std::endl);
    };
  }
  void createFramebuffers() {
    auto n = _swapChainImageViews.size();
    _swapChainFramebuffers.resize(n);
    for (int i = 0; i < n; (i) += (1)) {
      auto attachments = std::array<VkImageView, 2>(
          {_swapChainImageViews[i], _depthImageView});
      {
        VkFramebufferCreateInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        info.renderPass = _renderPass;
        info.attachmentCount = static_cast<uint32_t>(attachments.size());
        info.pAttachments = attachments.data();
        info.width = _swapChainExtent.width;
        info.height = _swapChainExtent.height;
        info.layers = 1;
        if (!((VK_SUCCESS) ==
              (vkCreateFramebuffer(_device, &info, nullptr,
                                   &(_swapChainFramebuffers[i]))))) {
          throw std::runtime_error(
              "failed to (vkCreateFramebuffer _device &info nullptr            "
              "(ref (aref _swapChainFramebuffers i)))");
        };
        (std::cout) << (std::chrono::high_resolution_clock::now()
                            .time_since_epoch()
                            .count())
                    << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                    << (__func__)
                    << (" create framebuffer (aref _swapChainFramebuffers i)=")
                    << (_swapChainFramebuffers[i]) << (std::endl);
      };
    };
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
    VkAttachmentDescription depthAttachment = {};
    depthAttachment.format = findDepthFormat();
    depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depthAttachment.finalLayout =
        VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    VkAttachmentReference colorAttachmentRef = {};
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    VkAttachmentReference depthAttachmentRef = {};
    depthAttachmentRef.attachment = 1;
    depthAttachmentRef.layout =
        VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    VkSubpassDescription subpass = {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentRef;
    subpass.pDepthStencilAttachment = &depthAttachmentRef;
    VkSubpassDependency dependency = {};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.srcAccessMask = 0;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.dstAccessMask = ((VK_ACCESS_COLOR_ATTACHMENT_READ_BIT) |
                                (VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT));
    auto attachments = std::array<VkAttachmentDescription, 2>(
        {colorAttachment, depthAttachment});
    {
      VkRenderPassCreateInfo info = {};
      info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
      info.attachmentCount = static_cast<uint32_t>(attachments.size());
      info.pAttachments = attachments.data();
      info.subpassCount = 1;
      info.pSubpasses = &subpass;
      info.dependencyCount = 1;
      info.pDependencies = &dependency;
      if (!((VK_SUCCESS) ==
            (vkCreateRenderPass(_device, &info, nullptr, &_renderPass)))) {
        throw std::runtime_error("failed to (vkCreateRenderPass _device &info "
                                 "nullptr &_renderPass)");
      };
      (std::cout) << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                  << (__func__) << (" create render-pass _renderPass=")
                  << (_renderPass) << (std::endl);
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
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
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
    VkPipelineDepthStencilStateCreateInfo depthStencil = {};
    depthStencil.sType =
        VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
    depthStencil.depthBoundsTestEnable = VK_FALSE;
    depthStencil.minDepthBounds = (0.0e+0f);
    depthStencil.maxDepthBounds = (1.e+0f);
    depthStencil.stencilTestEnable = VK_FALSE;
    depthStencil.front = {};
    depthStencil.back = {};
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
    {
      VkPipelineLayoutCreateInfo info = {};
      info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
      info.setLayoutCount = 1;
      info.pSetLayouts = &_descriptorSetLayout;
      info.pushConstantRangeCount = 0;
      info.pPushConstantRanges = nullptr;
      if (!((VK_SUCCESS) == (vkCreatePipelineLayout(_device, &info, nullptr,
                                                    &_pipelineLayout)))) {
        throw std::runtime_error("failed to (vkCreatePipelineLayout _device "
                                 "&info nullptr &_pipelineLayout)");
      };
      (std::cout) << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                  << (__func__) << (" create pipeline-layout _pipelineLayout=")
                  << (_pipelineLayout) << (std::endl);
    };
    {
      VkGraphicsPipelineCreateInfo info = {};
      info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
      info.stageCount = 2;
      info.pStages = shaderStages;
      info.pVertexInputState = &vertexInputInfo;
      info.pInputAssemblyState = &inputAssembly;
      info.pViewportState = &viewPortState;
      info.pRasterizationState = &rasterizer;
      info.pMultisampleState = &multisampling;
      info.pDepthStencilState = &depthStencil;
      info.pColorBlendState = &colorBlending;
      info.pDynamicState = nullptr;
      info.layout = _pipelineLayout;
      info.renderPass = _renderPass;
      info.subpass = 0;
      info.basePipelineHandle = VK_NULL_HANDLE;
      info.basePipelineIndex = -1;
      if (!((VK_SUCCESS) ==
            (vkCreateGraphicsPipelines(_device, VK_NULL_HANDLE, 1, &info,
                                       nullptr, &_graphicsPipeline)))) {
        throw std::runtime_error(
            "failed to (vkCreateGraphicsPipelines _device VK_NULL_HANDLE 1 "
            "&info nullptr            &_graphicsPipeline)");
      };
      (std::cout) << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                  << (__func__)
                  << (" create graphics-pipeline _graphicsPipeline=")
                  << (_graphicsPipeline) << (std::endl);
    };
    vkDestroyShaderModule(_device, fragShaderModule, nullptr);
    vkDestroyShaderModule(_device, vertShaderModule, nullptr);
  }
  VkShaderModule createShaderModule(const std::vector<char> &code) {
    VkShaderModule shaderModule;
    {
      VkShaderModuleCreateInfo info = {};
      info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
      info.codeSize = code.size();
      info.pCode = reinterpret_cast<const uint32_t *>(code.data());
      if (!((VK_SUCCESS) ==
            (vkCreateShaderModule(_device, &info, nullptr, &shaderModule)))) {
        throw std::runtime_error("failed to (vkCreateShaderModule _device "
                                 "&info nullptr &shaderModule)");
      };
      (std::cout) << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                  << (__func__) << (" create shader-module shaderModule=")
                  << (shaderModule) << (std::endl);
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
    {
      VkSwapchainCreateInfoKHR info = {};
      info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
      info.surface = _surface;
      info.minImageCount = imageCount;
      info.imageFormat = surfaceFormat.format;
      info.imageColorSpace = surfaceFormat.colorSpace;
      info.imageExtent = extent;
      info.imageArrayLayers = 1;
      info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
      info.imageSharingMode = imageSharingMode;
      info.queueFamilyIndexCount = queueFamilyIndexCount;
      info.pQueueFamilyIndices = pQueueFamilyIndices;
      info.preTransform = swapChainSupport.capabilities.currentTransform;
      info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
      info.presentMode = presentMode;
      info.clipped = VK_TRUE;
      info.oldSwapchain = VK_NULL_HANDLE;
      if (!((VK_SUCCESS) ==
            (vkCreateSwapchainKHR(_device, &info, nullptr, &_swapChain)))) {
        throw std::runtime_error("failed to (vkCreateSwapchainKHR _device "
                                 "&info nullptr &_swapChain)");
      };
      (std::cout) << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                  << (__func__) << (" create swapchain _swapChain=")
                  << (_swapChain) << (std::endl);
    };
    // now get the images, note will be destroyed with the swap chain
    vkGetSwapchainImagesKHR(_device, _swapChain, &imageCount, nullptr);
    _swapChainImages.resize(imageCount);
    vkGetSwapchainImagesKHR(_device, _swapChain, &imageCount,
                            _swapChainImages.data());
    _swapChainImageFormat = surfaceFormat.format;
    _swapChainExtent = extent;
  }
  VkImageView createImageView(VkImage image, VkFormat format,
                              VkImageAspectFlags aspectFlags,
                              uint32_t mipLevels) {
    VkImageView imageView;
    {
      VkImageViewCreateInfo info = {};
      info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
      info.image = image;
      info.viewType = VK_IMAGE_VIEW_TYPE_2D;
      info.format = format;
      info.subresourceRange.aspectMask = aspectFlags;
      info.subresourceRange.baseMipLevel = 0;
      info.subresourceRange.levelCount = mipLevels;
      info.subresourceRange.baseArrayLayer = 0;
      info.subresourceRange.layerCount = 1;
      if (!((VK_SUCCESS) ==
            (vkCreateImageView(_device, &info, nullptr, &imageView)))) {
        throw std::runtime_error(
            "failed to (vkCreateImageView _device &info nullptr &imageView)");
      };
      (std::cout) << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                  << (__func__) << (" create image-view imageView=")
                  << (imageView) << (std::endl);
    };
    return imageView;
  }
  void createImageViews() {
    _swapChainImageViews.resize(_swapChainImages.size());
    for (int i = 0; i < _swapChainImages.size(); (i) += (1)) {
      _swapChainImageViews[i] =
          createImageView(_swapChainImages[i], _swapChainImageFormat,
                          VK_IMAGE_ASPECT_COLOR_BIT, 1);
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
    deviceFeatures.samplerAnisotropy = VK_TRUE;
    {
      VkDeviceCreateInfo info = {};
      info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
      info.pQueueCreateInfos = queueCreateInfos.data();
      info.queueCreateInfoCount =
          static_cast<uint32_t>(queueCreateInfos.size());
      info.pEnabledFeatures = &deviceFeatures;
      info.enabledExtensionCount =
          static_cast<uint32_t>(_deviceExtensions.size());
      info.ppEnabledExtensionNames = _deviceExtensions.data();
      info.enabledLayerCount = static_cast<uint32_t>(_validationLayers.size());
      info.ppEnabledLayerNames = _validationLayers.data();
      if (!((VK_SUCCESS) ==
            (vkCreateDevice(_physicalDevice, &info, nullptr, &_device)))) {
        throw std::runtime_error("failed to (vkCreateDevice _physicalDevice "
                                 "&info nullptr &_device)");
      };
      (std::cout) << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                  << (__func__) << (" create device _physicalDevice=")
                  << (_physicalDevice) << (std::endl);
    };
    vkGetDeviceQueue(_device, indices.graphicsFamily.value(), 0,
                     &_graphicsQueue);
    vkGetDeviceQueue(_device, indices.presentFamily.value(), 0, &_presentQueue);
  }
  nil createColorResources() {
    VkFormat colorFormat;
    createImage(_swapChainExtent.width, _swapChainExtent.height, 1,
                _msaaSamples, colorFormat, VK_IMAGE_TILING_OPTIMAL,
                ((VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT) |
                 (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)),
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, _colorImage,
                _colorImageMemory);
    _colorImageView =
        createImageView(_colorImage, colorFormat, VK_IMAGE_LAYOUT_UNDEFINED,
                        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, 1);
  }
  VkSampleCountFlagBits getMaxUsableSampleCount() {
    VkPhysicalDeviceProperties physicalDeviceProperties;
    vkGetPhysicalDeviceProperties(_physicalDevice, &physicalDeviceProperties);
    auto count =
        std::min(physicalDeviceProperties.limits.framebufferColorSampleCounts,
                 physicalDeviceProperties.limits.framebufferDepthSampleCounts);
    if (((counts) & (VK_SAMPLE_COUNT_64_BIT))) {
      return VK_SAMPLE_COUNT_64_BIT;
    };
    if (((counts) & (VK_SAMPLE_COUNT_32_BIT))) {
      return VK_SAMPLE_COUNT_32_BIT;
    };
    if (((counts) & (VK_SAMPLE_COUNT_16_BIT))) {
      return VK_SAMPLE_COUNT_16_BIT;
    };
    if (((counts) & (VK_SAMPLE_COUNT_8_BIT))) {
      return VK_SAMPLE_COUNT_8_BIT;
    };
    if (((counts) & (VK_SAMPLE_COUNT_4_BIT))) {
      return VK_SAMPLE_COUNT_4_BIT;
    };
    if (((counts) & (VK_SAMPLE_COUNT_2_BIT))) {
      return VK_SAMPLE_COUNT_2_BIT;
    };
    return VK_SAMPLE_COUNT_1_BIT;
  };
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
        _msaaSamples = getMaxUsableSampleCount();
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
    const auto angularRate = glm::radians((9.e+0f));
    auto rotationAngle = ((time) * (angularRate));
    UniformBufferObject ubo = {};
    ubo.model = glm::rotate(glm::mat4((1.e+0f)), rotationAngle, zAxis);
    ubo.view = glm::lookAt(glm::vec3((2.e+0f), (2.e+0f), (2.e+0f)),
                           glm::vec3((0.0e+0f), (0.0e+0f), (0.0e+0f)), zAxis);
    ubo.proj = glm::perspective(
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
    (std::cout) << ("***** cleanupSwapChain") << (std::endl);
    (std::cout) << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                << (__func__) << (" cleanup depth: ") << (" _depthImageView=")
                << (_depthImageView) << (" _depthImage=") << (_depthImage)
                << (" _depthImageMemory=") << (_depthImageMemory)
                << (std::endl);
    vkDestroyImageView(_device, _depthImageView, nullptr);
    vkDestroyImage(_device, _depthImage, nullptr);
    vkFreeMemory(_device, _depthImageMemory, nullptr);
    for (auto &b : _swapChainFramebuffers) {
      (std::cout) << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                  << (__func__) << (" framebuffer: ") << (" b=") << (b)
                  << (std::endl);
      vkDestroyFramebuffer(_device, b, nullptr);
    };
    vkFreeCommandBuffers(_device, _commandPool,
                         static_cast<uint32_t>(_commandBuffers.size()),
                         _commandBuffers.data());
    (std::cout) << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                << (__func__) << (" pipeline: ") << (" _graphicsPipeline=")
                << (_graphicsPipeline) << (" _pipelineLayout=")
                << (_pipelineLayout) << (" _renderPass=") << (_renderPass)
                << (std::endl);
    vkDestroyPipeline(_device, _graphicsPipeline, nullptr);
    vkDestroyPipelineLayout(_device, _pipelineLayout, nullptr);
    vkDestroyRenderPass(_device, _renderPass, nullptr);
    for (auto &view : _swapChainImageViews) {
      (std::cout) << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                  << (__func__) << (" image-view: ") << (" view=") << (view)
                  << (std::endl);
      vkDestroyImageView(_device, view, nullptr);
    };
    (std::cout) << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                << (__func__) << (" swapchain: ") << (" _swapChain=")
                << (_swapChain) << (std::endl);
    vkDestroySwapchainKHR(_device, _swapChain, nullptr);
    for (int i = 0; i < _swapChainImages.size(); (i) += (1)) {
      (std::cout) << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                  << (__func__) << (" ubo: ") << (" (aref _uniformBuffers i)=")
                  << (_uniformBuffers[i])
                  << (" (aref _uniformBuffersMemory i)=")
                  << (_uniformBuffersMemory[i]) << (std::endl);
      vkDestroyBuffer(_device, _uniformBuffers[i], nullptr);
      vkFreeMemory(_device, _uniformBuffersMemory[i], nullptr);
    }
    (std::cout) << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                << (__func__) << (" descriptor-pool: ") << (" _descriptorPool=")
                << (_descriptorPool) << (std::endl);
    vkDestroyDescriptorPool(_device, _descriptorPool, nullptr);
  }
  void cleanup() {
    cleanupSwapChain();
    (std::cout) << ("***** cleanup") << (std::endl);
    (std::cout) << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                << (__func__) << (" tex: ") << (" _textureSampler=")
                << (_textureSampler) << (" _textureImageView=")
                << (_textureImageView) << (" _textureImage=") << (_textureImage)
                << (" _textureImageMemory=") << (_textureImageMemory)
                << (" _descriptorSetLayout=") << (_descriptorSetLayout)
                << (std::endl);
    vkDestroySampler(_device, _textureSampler, nullptr);
    vkDestroyImageView(_device, _textureImageView, nullptr);
    vkDestroyImage(_device, _textureImage, nullptr);
    vkFreeMemory(_device, _textureImageMemory, nullptr);
    vkDestroyDescriptorSetLayout(_device, _descriptorSetLayout, nullptr);
    (std::cout) << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                << (__func__) << (" buffers: ") << (" _vertexBuffer=")
                << (_vertexBuffer) << (" _vertexBufferMemory=")
                << (_vertexBufferMemory) << (" _indexBuffer=") << (_indexBuffer)
                << (" _indexBufferMemory=") << (_indexBufferMemory)
                << (std::endl);
    vkDestroyBuffer(_device, _vertexBuffer, nullptr);
    vkFreeMemory(_device, _vertexBufferMemory, nullptr);
    vkDestroyBuffer(_device, _indexBuffer, nullptr);
    vkFreeMemory(_device, _indexBufferMemory, nullptr);
    for (int i = 0; i < _MAX_FRAMES_IN_FLIGHT; (i) += (1)) {
      (std::cout) << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                  << (__func__) << (" sync: ")
                  << (" (aref _renderFinishedSemaphores i)=")
                  << (_renderFinishedSemaphores[i])
                  << (" (aref _imageAvailableSemaphores i)=")
                  << (_imageAvailableSemaphores[i])
                  << (" (aref _inFlightFences i)=") << (_inFlightFences[i])
                  << (std::endl);
      vkDestroySemaphore(_device, _renderFinishedSemaphores[i], nullptr);
      vkDestroySemaphore(_device, _imageAvailableSemaphores[i], nullptr);
      vkDestroyFence(_device, _inFlightFences[i], nullptr);
    }
    (std::cout) << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                << (__func__) << (" cmd-pool: ") << (" _commandPool=")
                << (_commandPool) << (std::endl);
    vkDestroyCommandPool(_device, _commandPool, nullptr);
    (std::cout) << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" ") << (__FILE__) << (":") << (__LINE__) << (" ")
                << (__func__) << (" rest: ") << (" _device=") << (_device)
                << (" _instance=") << (_instance) << (" _window=") << (_window)
                << (std::endl);
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
    (std::cerr) << (e.what()) << (std::endl);
    return EXIT_FAILURE;
  };
  return EXIT_SUCCESS;
}