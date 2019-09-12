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
    createRenderPass();
    createGraphicsPipeline();
    createFramebuffers();
    createCommandPool();
    createCommandBuffers();
    createSyncObjects();
  }
  void recreateSwapChain() {
    vkDeviceWaitIdle(_device);
    createSwapChain();
    createImageViews();
    createRenderPass();
    createGraphicsPipeline();
    createFramebuffers();
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
      if (!((((VK_SUCCESS) ==
              (vkCreateSemaphore(_device, &semaphoreInfo, nullptr,
                                 &(_imageAvailableSemaphores[i])))) &&
             ((VK_SUCCESS) ==
              (vkCreateSemaphore(_device, &semaphoreInfo, nullptr,
                                 &(_renderFinishedSemaphores[i])))) &&
             ((VK_SUCCESS) == (vkCreateFence(_device, &fenceInfo, nullptr,
                                             &(_inFlightFences[i]))))))) {
        throw std::runtime_error("failed to create sync objects.");
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
      throw std::runtime_error("failed to allocate command buffers.");
    };
    for (int i = 0; i < _commandBuffers.size(); (i) += (1)) {
      VkCommandBufferBeginInfo beginInfo = {};
      beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
      beginInfo.flags = 0;
      beginInfo.pInheritanceInfo = nullptr;
      if (!((VK_SUCCESS) ==
            (vkBeginCommandBuffer(_commandBuffers[i], &beginInfo)))) {
        throw std::runtime_error("failed to begin recording command buffer.");
      };
      VkClearValue clearColor = {(0.0e+0), (0.0e+0), (0.0e+0), (1.e+0)};
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
      vkCmdDraw(_commandBuffers[i], 3, 1, 0, 0);
      vkCmdEndRenderPass(_commandBuffers[i]);
      if (!((VK_SUCCESS) == (vkEndCommandBuffer(_commandBuffers[i])))) {
        throw std::runtime_error("failed to record command buffer.");
      };
    }
  }
  void createCommandPool() {
    auto queueFamilyIndices = findQueueFamilies(_physicalDevice);
    VkCommandPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();
    poolInfo.flags = 0;
    if (!((VK_SUCCESS) ==
          (vkCreateCommandPool(_device, &poolInfo, nullptr, &_commandPool)))) {
      throw std::runtime_error("failed to create command pool.");
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
        throw std::runtime_error("failed to create framebuffer.");
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
      throw std::runtime_error("failed to create render pass.");
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
    VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
    vertexInputInfo.sType =
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = 0;
    vertexInputInfo.pVertexBindingDescriptions = nullptr;
    vertexInputInfo.vertexAttributeDescriptionCount = 0;
    vertexInputInfo.pVertexAttributeDescriptions = nullptr;
    VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
    inputAssembly.sType =
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    inputAssembly.primitiveRestartEnable = VK_FALSE;
    VkViewport viewport = {};
    viewport.x = (0.0e+0);
    viewport.y = (0.0e+0);
    viewport.width = (((1.e+0)) * (_swapChainExtent.width));
    viewport.height = (((1.e+0)) * (_swapChainExtent.height));
    viewport.minDepth = (0.0e+0);
    viewport.maxDepth = (1.e+0);
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
    rasterizer.lineWidth = (1.e+0);
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
    rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
    rasterizer.depthBiasEnable = VK_FALSE;
    rasterizer.depthBiasConstantFactor = (0.0e+0);
    rasterizer.depthBiasClamp = (0.0e+0);
    rasterizer.depthBiasSlopeFactor = (0.0e+0);
    VkPipelineMultisampleStateCreateInfo multisampling = {};
    multisampling.sType =
        VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    multisampling.minSampleShading = (1.e+0);
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
    colorBlending.blendConstants[0] = (0.0e+0);
    colorBlending.blendConstants[1] = (0.0e+0);
    colorBlending.blendConstants[2] = (0.0e+0);
    colorBlending.blendConstants[3] = (0.0e+0);
    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 0;
    pipelineLayoutInfo.pSetLayouts = nullptr;
    pipelineLayoutInfo.pushConstantRangeCount = 0;
    pipelineLayoutInfo.pPushConstantRanges = nullptr;
    if (!((VK_SUCCESS) ==
          (vkCreatePipelineLayout(_device, &pipelineLayoutInfo, nullptr,
                                  &_pipelineLayout)))) {
      throw std::runtime_error("failed to create pipeline layout.");
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
      throw std::runtime_error("failed to create graphics pipeline.");
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
      drawFrame();
    }
    vkDeviceWaitIdle(_device);
  }
  void drawFrame() {
    vkWaitForFences(_device, 1, &(_inFlightFences[_currentFrame]), VK_TRUE,
                    UINT64_MAX);
    vkResetFences(_device, 1, &(_inFlightFences[_currentFrame]));
    uint32_t imageIndex = 0;
    vkAcquireNextImageKHR(_device, _swapChain, UINT64_MAX,
                          _imageAvailableSemaphores[_currentFrame],
                          VK_NULL_HANDLE, &imageIndex);
    VkSemaphore waitSemaphores[] = {_imageAvailableSemaphores[_currentFrame]};
    VkSemaphore signalSemaphores[] = {_renderFinishedSemaphores[_currentFrame]};
    VkPipelineStageFlags waitStages[] = {
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = waitSemaphores;
    submitInfo.pWaitDstStageMask = waitStages;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &(_commandBuffers[imageIndex]);
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = signalSemaphores;
    if (!((VK_SUCCESS) == (vkQueueSubmit(_graphicsQueue, 1, &submitInfo,
                                         _inFlightFences[_currentFrame])))) {
      throw std::runtime_error("failed to submit draw command buffer.");
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
    vkQueuePresentKHR(_presentQueue, &presentInfo);
    _currentFrame = ((1) + (_currentFrame)) % _MAX_FRAMES_IN_FLIGHT;
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
  }
  void cleanup() {
    cleanupSwapChain();
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