 
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
 
#include "utils.h"
 
#include "globals.h"
 
#include "proto2.h"
 ;
State state  = {._window=NULL, ._validationLayers={"VK_LAYER_KHRONOS_validation"}, ._physicalDevice=VK_NULL_HANDLE, ._deviceExtensions={VK_KHR_SWAPCHAIN_EXTENSION_NAME}};
void mainLoop (){
        {
                        __auto_type current_time  = now();
        printf("%6.6f", ((current_time)-(state._start_time)));
        printf(" ");
        printf(printf_dec_format(__FILE__), __FILE__);
        printf(":");
        printf(printf_dec_format(__LINE__), __LINE__);
        printf(" ");
        printf(printf_dec_format(__func__), __func__);
        printf(" mainLoop: ");
        printf("\n");
};
        while (!(glfwWindowShouldClose(state._window))) {
                glfwPollEvents();
                drawFrame();
}
        {
                        __auto_type current_time  = now();
        printf("%6.6f", ((current_time)-(state._start_time)));
        printf(" ");
        printf(printf_dec_format(__FILE__), __FILE__);
        printf(":");
        printf(printf_dec_format(__LINE__), __LINE__);
        printf(" ");
        printf(printf_dec_format(__func__), __func__);
        printf(" wait for gpu before cleanup: ");
        printf("\n");
};
        vkDeviceWaitIdle(state._device);
}
void run (){
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
}
int main (){
            state._start_time=now();
        run();
};