// generated with qwen2.5
#include <vulkan/vulkan.hpp>
#include <GLFW/glfw3.h>
#include <vector>
#include <cmath>
#include <iostream>

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

class SineCurveApp {
public:
    void run() {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

private:
    GLFWwindow* window{nullptr};
    vk::Instance instance;
    vk::PhysicalDevice physicalDevice;
    vk::Device device;
    vk::Queue graphicsQueue;
    VkSurfaceKHR_T* surface;
    vk::SwapchainKHR swapchain;
    std::vector<VkCommandBuffer> commandBuffers;
  
    void initWindow() {
        if (!glfwInit()) {
            throw std::runtime_error("Failed to initialize GLFW");
        }

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        window = glfwCreateWindow(WIDTH, HEIGHT, "Sine Curve Animation", nullptr, nullptr);
    }

    void initVulkan() {
        createInstance();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
    }

    void mainLoop() {
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            drawFrame();
        }
        vkDeviceWaitIdle(device);
    }

    void cleanup() {
        vkDestroySwapchainKHR(device, swapchain, nullptr);
//
//        for (size_t i = 0; i < commandBuffers.size(); i++) {
//            device.destroyCommandBuffer(commandBuffers[i]);
//        }
//
//        device.destroyCommandPool(commandPool);
//        device.destroyRenderPass(renderPass);
//        device.destroyPipelineLayout(pipelineLayout);
//        device.destroyGraphicsPipeline(graphicsPipeline);
//
//        device.destroySwapchainKHR(swapchain, nullptr);
//
//        for (size_t i = 0; i < swapChainImages.size(); i++) {
//            device.destroyImageView(swapChainImageViews[i]);
//        }
//
//        device.destroyDevice();
        vkDestroySurfaceKHR(instance, surface, nullptr);
//        vkDestroyInsta
//        nce(instance);

        glfwDestroyWindow(window);
        glfwTerminate();
    }

    void createInstance() {
        std::vector<vk::ExtensionProperties> extensions = vk::enumerateInstanceExtensionProperties();

        std::cout << "Available instance extensions:" << std::endl;
        for (const auto& extension : extensions) {
            std::cout << "\t" << std::string(extension.extensionName) << std::endl;
        }

        if (!glfwVulkanSupported()) {
            throw std::runtime_error("GLFW: Vulkan not supported");
        }
    }

    void createSurface() {
        if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create window surface");
        }
    }

    void pickPhysicalDevice() {
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

        if (deviceCount == 0) {
            throw std::runtime_error("Failed to find GPUs with Vulkan support");
        }

        std::vector<vk::PhysicalDevice> devices = vkEnumeratePhysicalDevices(instance, nullptr);

        for (const auto& device : devices) {
            if (isDeviceSuitable(device)) {
                physicalDevice = device;
                break;
            }
        }

        if (physicalDevice == VK_NULL_HANDLE) {
            throw std::runtime_error("Failed to find a suitable GPU");
        }
    }

    void createLogicalDevice() {
        queueFamilyIndices indices = findQueueFamilies(physicalDevice);

        vk::DeviceCreateInfo createInfo{};
        createInfo.queueCreateInfoCount = static_cast<uint32_t>(indices.graphicsFamily.value());
        createInfo.pQueueCreateInfos = indices.graphicsFamily.value();
        createInfo.enabledExtensionCount = 0;
        createInfo.enabledLayerCount = 0;

        device = physicalDevice.createDevice(createInfo, nullptr);

        graphicsQueue = device.getQueue(indices.graphicsFamily.value(), 0);
    }

    void drawFrame() {
        // Render the sine curve here
    }
};

int main() {
    SineCurveApp app;
    try {
        app.run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
