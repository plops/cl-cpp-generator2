 
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
 
#include "utils.h"
 
#include "globals.h"
 
#include "proto2.h"
 ;
extern State state;
bool hasStencilComponent (VkFormat format){
        return (((VK_FORMAT_D32_SFLOAT_S8_UINT)==(format))||((VK_FORMAT_D24_UNORM_S8_UINT)==(format)));
}
VkCommandBuffer beginSingleTimeCommands (){
            VkCommandBuffer commandBuffer ;
    {
                        VkCommandBufferAllocateInfo info  = {};
                info.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
                info.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY;
                info.commandPool=state._commandPool;
                info.commandBufferCount=1;
                        vkAllocateCommandBuffers(state._device, &info, &commandBuffer);
        {
                                    __auto_type current_time  = now();
            printf("%6.6f", ((current_time)-(state._start_time)));
            printf(" ");
            printf(printf_dec_format(__FILE__), __FILE__);
            printf(":");
            printf(printf_dec_format(__LINE__), __LINE__);
            printf(" ");
            printf(printf_dec_format(__func__), __func__);
            printf("  allocate command-buffer: ");
            printf("\n");
};
};
        {
                        VkCommandBufferBeginInfo info  = {};
                info.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
                info.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
                        vkBeginCommandBuffer(commandBuffer, &info);
        {
                                    __auto_type current_time  = now();
            printf("%6.6f", ((current_time)-(state._start_time)));
            printf(" ");
            printf(printf_dec_format(__FILE__), __FILE__);
            printf(":");
            printf(printf_dec_format(__LINE__), __LINE__);
            printf(" ");
            printf(printf_dec_format(__func__), __func__);
            printf("  begin command-buffer: ");
            printf(" commandBuffer=");
            printf(printf_dec_format(commandBuffer), commandBuffer);
            printf(" (%s)", type_string(commandBuffer));
            printf("\n");
};
};
        return commandBuffer;
}
void endSingleTimeCommands (VkCommandBuffer commandBuffer){
        vkEndCommandBuffer(commandBuffer);
            VkSubmitInfo submitInfo  = {};
        submitInfo.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount=1;
        submitInfo.pCommandBuffers=&commandBuffer;
        vkQueueSubmit(state._graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(state._graphicsQueue);
        vkFreeCommandBuffers(state._device, state._commandPool, 1, &commandBuffer);
        {
                        __auto_type current_time  = now();
        printf("%6.6f", ((current_time)-(state._start_time)));
        printf(" ");
        printf(printf_dec_format(__FILE__), __FILE__);
        printf(":");
        printf(printf_dec_format(__LINE__), __LINE__);
        printf(" ");
        printf(printf_dec_format(__func__), __func__);
        printf(" endSingleTimeCommands : ");
        printf(" commandBuffer=");
        printf(printf_dec_format(commandBuffer), commandBuffer);
        printf(" (%s)", type_string(commandBuffer));
        printf("\n");
};
}
void transitionImageLayout (VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout, uint32_t mipLevels){
            __auto_type commandBuffer  = beginSingleTimeCommands();
        VkImageMemoryBarrier barrier  = {};
        barrier.sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout=oldLayout;
        barrier.newLayout=newLayout;
        barrier.srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
        barrier.image=image;
        barrier.subresourceRange.aspectMask=VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.baseMipLevel=0;
        barrier.subresourceRange.levelCount=mipLevels;
        barrier.subresourceRange.baseArrayLayer=0;
        barrier.subresourceRange.layerCount=1;
        barrier.srcAccessMask=0;
        barrier.dstAccessMask=0;
    if ( (VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL)==(newLayout) ) {
                                barrier.subresourceRange.aspectMask=VK_IMAGE_ASPECT_DEPTH_BIT;
        if ( hasStencilComponent(format) ) {
                                                barrier.subresourceRange.aspectMask=((barrier.subresourceRange.aspectMask) | (VK_IMAGE_ASPECT_STENCIL_BIT));
};
} else {
                                barrier.subresourceRange.aspectMask=VK_IMAGE_ASPECT_COLOR_BIT;
}
        VkPipelineStageFlags srcStage ;
    VkPipelineStageFlags dstStage ;
    if ( (((VK_IMAGE_LAYOUT_UNDEFINED)==(oldLayout))&&((VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)==(newLayout))) ) {
                                barrier.srcAccessMask=0;
        barrier.dstAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;
        srcStage=VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        dstStage=VK_PIPELINE_STAGE_TRANSFER_BIT;
} else {
                if ( (((VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)==(oldLayout))&&((VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)==(newLayout))) ) {
                                                barrier.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask=VK_ACCESS_SHADER_READ_BIT;
            srcStage=VK_PIPELINE_STAGE_TRANSFER_BIT;
            dstStage=VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
} else {
                        if ( (((VK_IMAGE_LAYOUT_UNDEFINED)==(oldLayout))&&((VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL)==(newLayout))) ) {
                                                                barrier.srcAccessMask=0;
                barrier.dstAccessMask=((VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT) | (VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT));
                srcStage=VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
                dstStage=VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
} else {
                                if ( (((VK_IMAGE_LAYOUT_UNDEFINED)==(oldLayout))&&((VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)==(newLayout))) ) {
                                                                                barrier.srcAccessMask=0;
                    barrier.dstAccessMask=((VK_ACCESS_COLOR_ATTACHMENT_READ_BIT) | (VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT));
                    srcStage=VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
                    dstStage=VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
} else {
                                                            {
                                                                        __auto_type current_time  = now();
                        printf("%6.6f", ((current_time)-(state._start_time)));
                        printf(" ");
                        printf(printf_dec_format(__FILE__), __FILE__);
                        printf(":");
                        printf(printf_dec_format(__LINE__), __LINE__);
                        printf(" ");
                        printf(printf_dec_format(__func__), __func__);
                        printf(" unsupported layout transition.: ");
                        printf("\n");
};
}
}
}
};
    vkCmdPipelineBarrier(commandBuffer, srcStage, dstStage, 0, 0, NULL, 0, NULL, 1, &barrier);
    endSingleTimeCommands(commandBuffer);
}
uint32_t findMemoryType (uint32_t typeFilter, VkMemoryPropertyFlags properties){
            VkPhysicalDeviceMemoryProperties ps ;
    vkGetPhysicalDeviceMemoryProperties(state._physicalDevice, &ps);
    for (int i = 0;i<ps.memoryTypeCount;(i)+=(1)) {
                if ( (((((1)<<(i)) & (typeFilter)))&&((properties)==(((properties) & (ps.memoryTypes[i].propertyFlags))))) ) {
                                    return i;
};
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
        printf(" failed to find suitable memory type.: ");
        printf("\n");
};
}
 
Tuple_Image_DeviceMemory makeTuple_Image_DeviceMemory (VkImage image, VkDeviceMemory memory){
            Tuple_Image_DeviceMemory tup  = {image, memory};
    return tup;
}
Tuple_Image_DeviceMemory createImage (uint32_t width, uint32_t height, uint32_t mipLevels, VkSampleCountFlagBits numSamples, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties){
            VkImage image ;
    VkDeviceMemory imageMemory ;
    {
                        VkImageCreateInfo info  = {};
                info.sType=VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
                info.imageType=VK_IMAGE_TYPE_2D;
                info.extent.width=width;
                info.extent.height=height;
                info.extent.depth=1;
                info.mipLevels=mipLevels;
                info.arrayLayers=1;
                info.format=format;
                info.tiling=tiling;
                info.initialLayout=VK_IMAGE_LAYOUT_UNDEFINED;
                info.usage=usage;
                info.sharingMode=VK_SHARING_MODE_EXCLUSIVE;
                info.samples=numSamples;
                info.flags=0;
                        if ( !((VK_SUCCESS)==(vkCreateImage(state._device, &info, NULL, &image))) ) {
                                    {
                                                __auto_type current_time  = now();
                printf("%6.6f", ((current_time)-(state._start_time)));
                printf(" ");
                printf(printf_dec_format(__FILE__), __FILE__);
                printf(":");
                printf(printf_dec_format(__LINE__), __LINE__);
                printf(" ");
                printf(printf_dec_format(__func__), __func__);
                printf(" failed to (vkCreateImage (dot state _device) &info NULL &image): ");
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
            printf("  create image: ");
            printf(" image=");
            printf(printf_dec_format(image), image);
            printf(" (%s)", type_string(image));
            printf("\n");
};
};
            VkMemoryRequirements memReq ;
    vkGetImageMemoryRequirements(state._device, image, &memReq);
    {
                        VkMemoryAllocateInfo info  = {};
                info.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
                info.allocationSize=memReq.size;
                info.memoryTypeIndex=findMemoryType(memReq.memoryTypeBits, properties);
                        if ( !((VK_SUCCESS)==(vkAllocateMemory(state._device, &info, NULL, &imageMemory))) ) {
                                    {
                                                __auto_type current_time  = now();
                printf("%6.6f", ((current_time)-(state._start_time)));
                printf(" ");
                printf(printf_dec_format(__FILE__), __FILE__);
                printf(":");
                printf(printf_dec_format(__LINE__), __LINE__);
                printf(" ");
                printf(printf_dec_format(__func__), __func__);
                printf(" failed to (vkAllocateMemory (dot state _device) &info NULL &imageMemory): ");
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
            printf("  allocate memory: ");
            printf(" imageMemory=");
            printf(printf_dec_format(imageMemory), imageMemory);
            printf(" (%s)", type_string(imageMemory));
            printf("\n");
};
};
    vkBindImageMemory(state._device, image, imageMemory, 0);
    return makeTuple_Image_DeviceMemory(image, imageMemory);
}
void createColorResources (){
            VkFormat colorFormat  = state._swapChainImageFormat;
    __auto_type colorTuple  = createImage(state._swapChainExtent.width, state._swapChainExtent.height, 1, state._msaaSamples, colorFormat, VK_IMAGE_TILING_OPTIMAL, ((VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT) | (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)), VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        state._colorImage=colorTuple.image;
    state._colorImageMemory=colorTuple.memory;
        state._colorImageView=createImageView(state._colorImage, colorFormat, VK_IMAGE_ASPECT_COLOR_BIT, 1);
    transitionImageLayout(state._colorImage, colorFormat, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, 1);
};