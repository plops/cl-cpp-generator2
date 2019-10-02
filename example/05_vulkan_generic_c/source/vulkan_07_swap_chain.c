 
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
 
#include "utils.h"
 
#include "globals.h"
 
#include "proto2.h"
 ;
extern State state;
#include <stdlib.h>
#include <assert.h>
VkSurfaceFormatKHR chooseSwapSurfaceFormat (const VkSurfaceFormatKHR* availableFormats, int n){
        for (int i = 0;i<n;(i)+=(1)) {
                        __auto_type format  = availableFormats[i];
        if ( (((VK_FORMAT_B8G8R8A8_UNORM)==(format.format))&&((VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)==(format.colorSpace))) ) {
                                    return format;
};
}
        return availableFormats[0];
}
VkPresentModeKHR chooseSwapPresentMode (const VkPresentModeKHR* modes, int n){
        // prefer triple buffer (if available)
        for (int i = 0;i<n;(i)+=(1)) {
                        __auto_type mode  = modes[i];
        if ( (VK_PRESENT_MODE_MAILBOX_KHR)==(mode) ) {
                                    return mode;
};
}
        return VK_PRESENT_MODE_FIFO_KHR;
}
VkExtent2D chooseSwapExtent (const VkSurfaceCapabilitiesKHR* capabilities){
        if ( (UINT32_MAX)!=(capabilities->currentExtent.width) ) {
                        return capabilities->currentExtent;
} else {
                                int width  = 0;
        int height  = 0;
        glfwGetFramebufferSize(state._window, &width, &height);
                VkExtent2D actualExtent  = {width, height};
                actualExtent.width=max(capabilities->minImageExtent.width, min(capabilities->maxImageExtent.width, actualExtent.width));
                actualExtent.height=max(capabilities->minImageExtent.height, min(capabilities->maxImageExtent.height, actualExtent.height));
        return actualExtent;
}
};
void createSwapChain (){
            __auto_type swapChainSupport  = querySwapChainSupport(state._physicalDevice);
    __auto_type surfaceFormat  = chooseSwapSurfaceFormat(swapChainSupport.formats, swapChainSupport.formatsCount);
    __auto_type presentMode  = chooseSwapPresentMode(swapChainSupport.presentModes, swapChainSupport.presentModesCount);
    __auto_type extent  = chooseSwapExtent(&swapChainSupport.capabilities);
    __auto_type imageCount_  = ((swapChainSupport.capabilities.minImageCount)+(1));
    __auto_type imageCount  = max(imageCount_, _N_IMAGES);
    __auto_type indices  = findQueueFamilies(state._physicalDevice);
    __typeof__(indices.graphicsFamily) queueFamilyIndices[]  = {indices.graphicsFamily, indices.presentFamily};
    __auto_type imageSharingMode  = VK_SHARING_MODE_EXCLUSIVE;
    __auto_type queueFamilyIndexCount  = 0;
    __auto_type pQueueFamilyIndices  = NULL;
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
        printf(" create swap chain: ");
        printf(" imageCount_=");
        printf(printf_dec_format(imageCount_), imageCount_);
        printf(" (%s)", type_string(imageCount_));
        printf(" imageCount=");
        printf(printf_dec_format(imageCount), imageCount);
        printf(" (%s)", type_string(imageCount));
        printf(" swapChainSupport.capabilities.minImageCount=");
        printf(printf_dec_format(swapChainSupport.capabilities.minImageCount), swapChainSupport.capabilities.minImageCount);
        printf(" (%s)", type_string(swapChainSupport.capabilities.minImageCount));
        printf(" swapChainSupport.capabilities.maxImageCount=");
        printf(printf_dec_format(swapChainSupport.capabilities.maxImageCount), swapChainSupport.capabilities.maxImageCount);
        printf(" (%s)", type_string(swapChainSupport.capabilities.maxImageCount));
        printf("\n");
};
    if ( !((0)==(swapChainSupport.capabilities.maxImageCount)) ) {
                        assert((imageCount)<=(swapChainSupport.capabilities.maxImageCount));
};
    if ( !((indices.presentFamily)==(indices.graphicsFamily)) ) {
                        // this could be improved with ownership stuff
                imageSharingMode=VK_SHARING_MODE_CONCURRENT;
        queueFamilyIndexCount=2;
        pQueueFamilyIndices=pQueueFamilyIndices;
};
    if ( ((0<swapChainSupport.capabilities.maxImageCount)&&(swapChainSupport.capabilities.maxImageCount<imageCount)) ) {
                                imageCount=swapChainSupport.capabilities.maxImageCount;
};
    {
                        VkSwapchainCreateInfoKHR info  = {};
                info.sType=VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
                info.surface=state._surface;
                info.minImageCount=imageCount;
                info.imageFormat=surfaceFormat.format;
                info.imageColorSpace=surfaceFormat.colorSpace;
                info.imageExtent=extent;
                info.imageArrayLayers=1;
                info.imageUsage=VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
                info.imageSharingMode=imageSharingMode;
                info.queueFamilyIndexCount=queueFamilyIndexCount;
                info.pQueueFamilyIndices=pQueueFamilyIndices;
                info.preTransform=swapChainSupport.capabilities.currentTransform;
                info.compositeAlpha=VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
                info.presentMode=presentMode;
                info.clipped=VK_TRUE;
                info.oldSwapchain=VK_NULL_HANDLE;
                        if ( !((VK_SUCCESS)==(vkCreateSwapchainKHR(state._device, &info, NULL, &(state._swapChain)))) ) {
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
                printf(" failed to (vkCreateSwapchainKHR (dot state _device) &info NULL            (ref (dot state _swapChain))): ");
                printf("\n");
};
};
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
            printf("  create swapchain: ");
            printf(" state._swapChain=");
            printf(printf_dec_format(state._swapChain), state._swapChain);
            printf(" (%s)", type_string(state._swapChain));
            printf("\n");
};
};
    cleanupSwapChainSupport(&swapChainSupport);
        // now get the images, note will be destroyed with the swap chain
    vkGetSwapchainImagesKHR(state._device, state._swapChain, &imageCount, NULL);
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
        printf(" create swapChainImages: ");
        printf(" imageCount=");
        printf(printf_dec_format(imageCount), imageCount);
        printf(" (%s)", type_string(imageCount));
        printf(" _N_IMAGES=");
        printf(printf_dec_format(_N_IMAGES), _N_IMAGES);
        printf(" (%s)", type_string(_N_IMAGES));
        printf("\n");
};
    vkGetSwapchainImagesKHR(state._device, state._swapChain, &imageCount, state._swapChainImages);
        state._swapChainImageFormat=surfaceFormat.format;
    state._swapChainExtent=extent;
};