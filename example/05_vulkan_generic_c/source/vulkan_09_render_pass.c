 
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
 
#include "utils.h"
 
#include "globals.h"
 
#include "proto2.h"
 ;
extern State state;
void cleanupRenderPass (){
}
VkFormat findSupportedFormat (VkFormat* candidates, int n, VkImageTiling tiling, VkFormatFeatureFlags features){
        for (int i = 0;i<n;(i)+=(1)) {
                        __auto_type format  = candidates[i];
        VkFormatProperties props ;
        vkGetPhysicalDeviceFormatProperties(state._physicalDevice, format, &props);
        if ( (((VK_IMAGE_TILING_LINEAR)==(tiling))&&((features)==(((features) & (props.linearTilingFeatures))))) ) {
                                    return format;
};
        if ( (((VK_IMAGE_TILING_OPTIMAL)==(tiling))&&((features)==(((features) & (props.optimalTilingFeatures))))) ) {
                                    return format;
};
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
        printf(" failed to find supported format!: ");
        printf("\n");
};
}
VkFormat findDepthFormat (){
            VkFormat candidates[]  = {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT};
    return findSupportedFormat(candidates, length(candidates), VK_IMAGE_TILING_OPTIMAL, VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
}
void createRenderPass (){
            VkAttachmentDescription colorAttachment  = {};
        colorAttachment.format=state._swapChainImageFormat;
        colorAttachment.samples=state._msaaSamples;
        colorAttachment.loadOp=VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp=VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.stencilLoadOp=VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp=VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAttachment.initialLayout=VK_IMAGE_LAYOUT_UNDEFINED;
        colorAttachment.finalLayout=VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            VkAttachmentDescription depthAttachment  = {};
        depthAttachment.format=findDepthFormat();
        depthAttachment.samples=state._msaaSamples;
        depthAttachment.loadOp=VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAttachment.storeOp=VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttachment.stencilLoadOp=VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        depthAttachment.stencilStoreOp=VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttachment.initialLayout=VK_IMAGE_LAYOUT_UNDEFINED;
        depthAttachment.finalLayout=VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            VkAttachmentDescription colorAttachmentResolve  = {};
        colorAttachmentResolve.format=state._swapChainImageFormat;
        colorAttachmentResolve.samples=VK_SAMPLE_COUNT_1_BIT;
        colorAttachmentResolve.loadOp=VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachmentResolve.storeOp=VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachmentResolve.stencilLoadOp=VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachmentResolve.stencilStoreOp=VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAttachmentResolve.initialLayout=VK_IMAGE_LAYOUT_UNDEFINED;
        colorAttachmentResolve.finalLayout=VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
            VkAttachmentReference colorAttachmentRef  = {};
        colorAttachmentRef.attachment=0;
        colorAttachmentRef.layout=VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            VkAttachmentReference depthAttachmentRef  = {};
        depthAttachmentRef.attachment=1;
        depthAttachmentRef.layout=VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            VkAttachmentReference colorAttachmentResolveRef  = {};
        colorAttachmentResolveRef.attachment=2;
        colorAttachmentResolveRef.layout=VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            VkSubpassDescription subpass  = {};
        subpass.pipelineBindPoint=VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount=1;
        subpass.pColorAttachments=&colorAttachmentRef;
        subpass.pDepthStencilAttachment=&depthAttachmentRef;
        subpass.pResolveAttachments=&colorAttachmentResolveRef;
            VkSubpassDependency dependency  = {};
        dependency.srcSubpass=VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass=0;
        dependency.srcStageMask=VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.srcAccessMask=0;
        dependency.dstStageMask=VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.dstAccessMask=((VK_ACCESS_COLOR_ATTACHMENT_READ_BIT) | (VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT));
            VkAttachmentDescription attachments[]  = {colorAttachment, depthAttachment, colorAttachmentResolve};
    {
                        VkRenderPassCreateInfo info  = {};
                info.sType=VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
                info.attachmentCount=length(attachments);
                info.pAttachments=attachments;
                info.subpassCount=1;
                info.pSubpasses=&subpass;
                info.dependencyCount=1;
                info.pDependencies=&dependency;
                        if ( !((VK_SUCCESS)==(vkCreateRenderPass(state._device, &info, NULL, &(state._renderPass)))) ) {
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
                printf(" failed to (vkCreateRenderPass (dot state _device) &info NULL            (ref (dot state _renderPass))): ");
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
            printf("  create render-pass: ");
            printf(" state._renderPass=");
            printf(printf_dec_format(state._renderPass), state._renderPass);
            printf(" (%s)", type_string(state._renderPass));
            printf("\n");
};
};
};