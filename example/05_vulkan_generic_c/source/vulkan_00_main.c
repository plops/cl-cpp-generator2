 
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
 
#include "utils.h"
 
#include "globals.h"
 
#include "proto2.h"
 ;
State state  = {._window=NULL, ._validationLayers={"VK_LAYER_KHRONOS_validation"}, ._physicalDevice=VK_NULL_HANDLE, ._deviceExtensions={VK_KHR_SWAPCHAIN_EXTENSION_NAME}};
void mainLoop (){
        {
                        struct timespec tp ;
        clock_gettime(CLOCK_REALTIME, &tp);
        printf(printf_dec_format(tp.tv_sec), tp.tv_sec);
        printf(".");
        printf(printf_dec_format(tp.tv_nsec), tp.tv_nsec);
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
                        struct timespec tp ;
        clock_gettime(CLOCK_REALTIME, &tp);
        printf(printf_dec_format(tp.tv_sec), tp.tv_sec);
        printf(".");
        printf(printf_dec_format(tp.tv_nsec), tp.tv_nsec);
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
        run();
};