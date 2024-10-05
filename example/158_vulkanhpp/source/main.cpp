#include <vulkan/vulkan.hpp>
#include <GLFW/glfw3.h>
#include <format>
#include <iostream>
#include <vector>
#include <array>
#include <list>
#include <queue>
#include <string>

int main(int argc, char **argv) {

//### 3. Initialize GLFW


    if (!glfwInit()) {
        throw std::runtime_error("Failed to initialize GLFW");
    }

// Create a windowed mode window and its OpenGL context
    GLFWwindow *window = glfwCreateWindow(800, 600, "Vulkan Window", nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        throw std::runtime_error("Failed to create GLFW window");
    }
    glfwMakeContextCurrent(window);

// Enable VSync (optional)
    glfwSwapInterval(1);

// ### 4. Create Vulkan Instance
// - **Create a Vulkan instance**: This is the entry point to all Vulkan functionality.

    auto appInfo =
            vk::ApplicationInfo()
                    .setPApplicationName("Hello Triangle")
                    .setApplicationVersion(VK_MAKE_VERSION(1, 0, 0))
                    .setPEngineName("No Engine")
                    .setEngineVersion(VK_MAKE_VERSION(1, 0, 0))
                    .setApiVersion(VK_API_VERSION_1_2);

    auto eprops = vk::enumerateInstanceExtensionProperties();
    std::vector<const char *> seprops(eprops.size());
    for (const auto &eprop: eprops) {
        const char *s{eprop.extensionName};
        seprops.push_back(s);
        std::cout << s << std::endl;
    }
    auto createInfo =
            vk::InstanceCreateInfo()
                    .setPApplicationInfo(&appInfo)
                    .setPEnabledExtensionNames(seprops)
//                    .setPEnabledLayerNames(vk::enumerateInstanceLayerProperties());
//
//    auto instance = vk::createInstance(createInfo);
//
//
//// ### 5. Create a Surface
//// - **Create a surface**: A surface is required to present images on the screen.
//
//    vk::SurfaceKHR surface(instance, reinterpret_cast<VkSurfaceKHR>(glfwGetWindowSurface(window)));
//
//
//// ### 6. Select Physical Device
//// - **Select a physical device**: Choose a GPU that supports Vulkan and has the necessary extensions and queues.
//
//    auto devices = instance.enumeratePhysicalDevices().value();
//    vk::PhysicalDevice physicalDevice;
//
//    for (const auto &device: devices) {
//        if (checkDeviceSuitability(device)) {
//            physicalDevice = device;
//            break;
//        }
//    }
//
//    if (!physicalDevice) {
//        throw std::runtime_error("Failed to find a suitable GPU");
//    }
//

// ### 7. Create Logical Device

    return 0;
}
