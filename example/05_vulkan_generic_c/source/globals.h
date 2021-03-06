#ifndef GLOBALS_H
 
#define GLOBALS_H
 
enum {_N_IMAGES=4,_MAX_FRAMES_IN_FLIGHT=2};
struct State {
        double _start_time;
        GLFWwindow* _window;
        VkInstance _instance;
        VkPhysicalDevice _physicalDevice;
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
        VkQueue _presentQueue;
        VkSurfaceKHR _surface;
        const char* const _deviceExtensions[1];
        VkSwapchainKHR _swapChain;
        VkImage _swapChainImages[_N_IMAGES];
        VkFormat _swapChainImageFormat;
        VkExtent2D _swapChainExtent;
        VkImageView _swapChainImageViews[_N_IMAGES];
        VkDescriptorSetLayout _descriptorSetLayout;
        VkPipelineLayout _pipelineLayout;
        VkRenderPass _renderPass;
        VkPipeline _graphicsPipeline;
        VkFramebuffer _swapChainFramebuffers[_N_IMAGES];
        VkCommandPool _commandPool;
        VkCommandBuffer _commandBuffers[_N_IMAGES];
        VkSemaphore _imageAvailableSemaphores[_MAX_FRAMES_IN_FLIGHT];
        VkSemaphore _renderFinishedSemaphores[_MAX_FRAMES_IN_FLIGHT];
        size_t _currentFrame;
        VkFence _inFlightFences[_MAX_FRAMES_IN_FLIGHT];
        _Bool _framebufferResized;
        VkBuffer _vertexBuffer;
        VkBuffer _indexBuffer;
        VkDeviceMemory _vertexBufferMemory;
        VkDeviceMemory _indexBufferMemory;
        VkBuffer _uniformBuffers[_N_IMAGES];
        VkDeviceMemory _uniformBuffersMemory[_N_IMAGES];
        VkDescriptorPool _descriptorPool;
        VkDescriptorSet _descriptorSets[_N_IMAGES];
        VkImage _depthImage;
        VkDeviceMemory _depthImageMemory;
        VkImageView _depthImageView;
        Vertex* _vertices;
        int _num_vertices;
        uint32_t* _indices;
        int _num_indices;
};
typedef struct State State;
 
#endif
 