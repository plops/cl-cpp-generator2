 
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
 
#include "utils.h"
 
#include "globals.h"
 
#include "proto2.h"
 ;
extern State state;
#include <stdlib.h>
#include <string.h>
void cleanupPhysicalDevice (){
}
 
_Bool QueueFamilyIndices_isComplete (QueueFamilyIndices q){
        return (((-1)!=(q.graphicsFamily))&&((-1)!=(q.presentFamily)));
}
QueueFamilyIndices QueueFamilyIndices_make (){
            QueueFamilyIndices q ;
        q.graphicsFamily=-1;
    q.presentFamily=-1;
    return q;
}
void QueueFamilyIndices_destroy (QueueFamilyIndices* q){
        free(q);
}
QueueFamilyIndices findQueueFamilies (VkPhysicalDevice device){
            __auto_type indices  = QueueFamilyIndices_make();
    uint32_t queueFamilyCount  = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, NULL);
        VkQueueFamilyProperties queueFamilies[queueFamilyCount] ;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies);
        __auto_type i  = 0;
    for (int family_idx = 0;family_idx<((sizeof(queueFamilies))/(sizeof(*(queueFamilies))));(family_idx)+=(1)) {
                        __auto_type family  = queueFamilies[family_idx];
        {
                        if ( ((0<family.queueCount)&&(((family.queueFlags) & (VK_QUEUE_GRAPHICS_BIT)))) ) {
                                                                indices.graphicsFamily=i;
};
                                    VkBool32 presentSupport  = false;
            vkGetPhysicalDeviceSurfaceSupportKHR(device, i, state._surface, &presentSupport);
            if ( ((0<family.queueCount)&&(presentSupport)) ) {
                                                                indices.presentFamily=i;
};
            if ( QueueFamilyIndices_isComplete(indices) ) {
                                                break;
};
                        (i)++;
};
};
    return indices;
};
 
void cleanupSwapChainSupport (SwapChainSupportDetails* details){
        free(details->formats);
        free(details->presentModes);
            details->formatsCount=0;
    details->presentModesCount=0;
}
SwapChainSupportDetails querySwapChainSupport (VkPhysicalDevice device){
            SwapChainSupportDetails details  = {.formatsCount=0, .presentModesCount=0};
    __auto_type s  = state._surface;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, s, &details.capabilities);
        uint32_t formatCount  = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, s, &formatCount, NULL);
    if ( !((0)==(formatCount)) ) {
                                details.formatsCount=formatCount;
        details.formats=malloc(((sizeof(VkSurfaceFormatKHR))*(formatCount)));
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, s, &formatCount, details.formats);
};
        uint32_t presentModeCount  = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, s, &presentModeCount, NULL);
    if ( !((0)==(presentModeCount)) ) {
                                details.presentModesCount=presentModeCount;
        details.presentModes=(VkPresentModeKHR*)(malloc(((sizeof(VkPresentModeKHR))*(presentModeCount))));
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, s, &presentModeCount, details.presentModes);
};
    return details;
};
bool isDeviceSuitable (VkPhysicalDevice device){
            __auto_type extensionsSupported  = checkDeviceExtensionSupport(device);
    bool swapChainAdequate  = false;
    if ( extensionsSupported ) {
                                __auto_type swapChainSupport  = querySwapChainSupport(device);
                swapChainAdequate=((!((0)==(swapChainSupport.formatsCount)))&&(!((0)==(swapChainSupport.presentModesCount))));
        cleanupSwapChainSupport(&swapChainSupport);
};
            __auto_type indices  = findQueueFamilies(device);
    VkPhysicalDeviceFeatures supportedFeatures ;
    vkGetPhysicalDeviceFeatures(device, &supportedFeatures);
        __auto_type res  = ((QueueFamilyIndices_isComplete(indices))&&(supportedFeatures.samplerAnisotropy)&&(((extensionsSupported)&&(swapChainAdequate))));
    return res;
}
bool checkDeviceExtensionSupport (VkPhysicalDevice device){
            uint32_t extensionCount  = 0;
    vkEnumerateDeviceExtensionProperties(device, NULL, &extensionCount, NULL);
        VkExtensionProperties availableExtensions[extensionCount] ;
    vkEnumerateDeviceExtensionProperties(device, NULL, &extensionCount, availableExtensions);
    for (int required_idx = 0;required_idx<((sizeof(state._deviceExtensions))/(sizeof(*(state._deviceExtensions))));(required_idx)+=(1)) {
                        __auto_type required  = state._deviceExtensions[required_idx];
        {
                                    bool found  = false;
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
                printf(" check for extension: ");
                printf(" required=");
                printf(printf_dec_format(required), required);
                printf(" (%s)", type_string(required));
                printf("\n");
};
            for (int extension_idx = 0;extension_idx<((sizeof(availableExtensions))/(sizeof(*(availableExtensions))));(extension_idx)+=(1)) {
                                                __auto_type extension  = availableExtensions[extension_idx];
                {
                                        if ( (0)==(strcmp(extension.extensionName, required)) ) {
                                                                                                found=true;
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
                            printf(" check for extension: ");
                            printf(" found=");
                            printf(printf_dec_format(found), found);
                            printf(" (%s)", type_string(found));
                            printf("\n");
};
                        break;
};
};
};
            if ( !(found) ) {
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
                    printf(" not all of the required extensions were found: ");
                    printf(" required=");
                    printf(printf_dec_format(required), required);
                    printf(" (%s)", type_string(required));
                    printf(" found=");
                    printf(printf_dec_format(found), found);
                    printf(" (%s)", type_string(found));
                    printf("\n");
};
                return false;
};
};
};
    return true;
}
VkSampleCountFlagBits getMaxUsableSampleCount (){
            VkPhysicalDeviceProperties physicalDeviceProperties ;
    vkGetPhysicalDeviceProperties(state._physicalDevice, &physicalDeviceProperties);
        __auto_type count  = min(physicalDeviceProperties.limits.framebufferColorSampleCounts, physicalDeviceProperties.limits.framebufferDepthSampleCounts);
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
        printf(" min: ");
        printf(" count=");
        printf(printf_dec_format(count), count);
        printf(" (%s)", type_string(count));
        printf(" physicalDeviceProperties.limits.framebufferColorSampleCounts=");
        printf(printf_dec_format(physicalDeviceProperties.limits.framebufferColorSampleCounts), physicalDeviceProperties.limits.framebufferColorSampleCounts);
        printf(" (%s)", type_string(physicalDeviceProperties.limits.framebufferColorSampleCounts));
        printf(" physicalDeviceProperties.limits.framebufferDepthSampleCounts=");
        printf(printf_dec_format(physicalDeviceProperties.limits.framebufferDepthSampleCounts), physicalDeviceProperties.limits.framebufferDepthSampleCounts);
        printf(" (%s)", type_string(physicalDeviceProperties.limits.framebufferDepthSampleCounts));
        printf("\n");
};
    if ( ((count) & (VK_SAMPLE_COUNT_64_BIT)) ) {
                        return VK_SAMPLE_COUNT_64_BIT;
};
    if ( ((count) & (VK_SAMPLE_COUNT_32_BIT)) ) {
                        return VK_SAMPLE_COUNT_32_BIT;
};
    if ( ((count) & (VK_SAMPLE_COUNT_16_BIT)) ) {
                        return VK_SAMPLE_COUNT_16_BIT;
};
    if ( ((count) & (VK_SAMPLE_COUNT_8_BIT)) ) {
                        return VK_SAMPLE_COUNT_8_BIT;
};
    if ( ((count) & (VK_SAMPLE_COUNT_4_BIT)) ) {
                        return VK_SAMPLE_COUNT_4_BIT;
};
    if ( ((count) & (VK_SAMPLE_COUNT_2_BIT)) ) {
                        return VK_SAMPLE_COUNT_2_BIT;
};
    return VK_SAMPLE_COUNT_1_BIT;
}
void pickPhysicalDevice (){
        // initialize member _physicalDevice
            uint32_t deviceCount  = 0;
    vkEnumeratePhysicalDevices(state._instance, &deviceCount, NULL);
    if ( (0)==(deviceCount) ) {
                        // throw
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
            printf(" failed to find gpu with vulkan support.: ");
            printf("\n");
};
};
        VkPhysicalDevice devices[deviceCount] ;
    vkEnumeratePhysicalDevices(state._instance, &deviceCount, devices);
    for (int device_idx = 0;device_idx<((sizeof(devices))/(sizeof(*(devices))));(device_idx)+=(1)) {
                        __auto_type device  = devices[device_idx];
        {
                        if ( isDeviceSuitable(device) ) {
                                                                state._physicalDevice=device;
                state._msaaSamples=getMaxUsableSampleCount();
                break;
};
};
};
    if ( (VK_NULL_HANDLE)==(state._physicalDevice) ) {
                        // throw
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
            printf(" failed to find a suitable gpu.: ");
            printf("\n");
};
};
};