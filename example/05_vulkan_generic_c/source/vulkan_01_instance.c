 
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
 
#include "utils.h"
 
#include "globals.h"
 
#include "proto2.h"
 ;
extern State state;
#include <string.h>
void cleanupInstance (){
        vkDestroyInstance(state._instance, NULL);
}
void createInstance (){
        // initialize member _instance
            VkApplicationInfo appInfo  = {};
        appInfo.sType=VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName="Hello Triangle";
        appInfo.applicationVersion=VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName="No Engine";
        appInfo.engineVersion=VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion=VK_API_VERSION_1_0;
            uint32_t glfwExtensionCount  = 0;
    const char** glfwExtensions ;
        glfwExtensions=glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
    {
                        VkInstanceCreateInfo info  = {};
                info.sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
                info.pApplicationInfo=&appInfo;
                info.enabledExtensionCount=glfwExtensionCount;
                info.ppEnabledExtensionNames=glfwExtensions;
                info.enabledLayerCount=0;
                info.ppEnabledLayerNames=NULL;
                        if ( !((VK_SUCCESS)==(vkCreateInstance(&info, NULL, &(state._instance)))) ) {
                                    {
                                                __auto_type current_time  = now();
                printf("%6.6f", ((current_time)-(state._start_time)));
                printf(" ");
                printf(printf_dec_format(__FILE__), __FILE__);
                printf(":");
                printf(printf_dec_format(__LINE__), __LINE__);
                printf(" ");
                printf(printf_dec_format(__func__), __func__);
                printf(" failed to (vkCreateInstance &info NULL (ref (dot state _instance))): ");
                printf("\n");
};
};
        {
                                    __auto_type current_time  = now();
            printf("%6.6f", ((current_time)-(state._start_time)));
            printf(" ");
            printf(printf_dec_format(__FILE__), __FILE__);
            printf(":");
            printf(printf_dec_format(__LINE__), __LINE__);
            printf(" ");
            printf(printf_dec_format(__func__), __func__);
            printf("  create instance: ");
            printf(" state._instance=");
            printf(printf_dec_format(state._instance), state._instance);
            printf(" (%s)", type_string(state._instance));
            printf("\n");
};
};
};