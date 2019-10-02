 
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
_Bool checkValidationLayerSupport (){
            uint32_t layerCount  = 0;
    vkEnumerateInstanceLayerProperties(&layerCount, NULL);
        VkLayerProperties availableLayers[layerCount] ;
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers);
    for (int layerName_idx = 0;layerName_idx<((sizeof(state._validationLayers))/(sizeof(*(state._validationLayers))));(layerName_idx)+=(1)) {
                        __auto_type layerName  = state._validationLayers[layerName_idx];
        {
                                    __auto_type layerFound  = false;
            for (int layerProperties_idx = 0;layerProperties_idx<((sizeof(availableLayers))/(sizeof(*(availableLayers))));(layerProperties_idx)+=(1)) {
                                                __auto_type layerProperties  = availableLayers[layerProperties_idx];
                {
                                        if ( (0)==(strcmp(layerName, layerProperties.layerName)) ) {
                                                                        {
                                                                                    __auto_type current_time  = now();
                            printf("%6.6f", ((current_time)-(state._start_time)));
                            printf(" ");
                            printf(printf_dec_format(__FILE__), __FILE__);
                            printf(":");
                            printf(printf_dec_format(__LINE__), __LINE__);
                            printf(" ");
                            printf(printf_dec_format(__func__), __func__);
                            printf(" look for layer: ");
                            printf(" layerName=");
                            printf(printf_dec_format(layerName), layerName);
                            printf(" (%s)", type_string(layerName));
                            printf("\n");
};
                                                layerFound=true;
                        break;
};
};
};
            if ( !(layerFound) ) {
                                                return false;
};
};
};
    return true;
}
void createInstance (){
        // initialize member _instance
        if ( !(checkValidationLayerSupport()) ) {
                        // throw
        {
                                    __auto_type current_time  = now();
            printf("%6.6f", ((current_time)-(state._start_time)));
            printf(" ");
            printf(printf_dec_format(__FILE__), __FILE__);
            printf(":");
            printf(printf_dec_format(__LINE__), __LINE__);
            printf(" ");
            printf(printf_dec_format(__func__), __func__);
            printf(" validation layers requested, but unavailable.: ");
            printf("\n");
};
};
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
                info.enabledLayerCount=length(state._validationLayers);
                info.ppEnabledLayerNames=state._validationLayers;
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