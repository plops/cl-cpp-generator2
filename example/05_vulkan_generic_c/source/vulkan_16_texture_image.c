 
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
 
#include "utils.h"
 
#include "globals.h"
 
#include "proto2.h"
 ;
extern State state;
#pragma GCC optimize ("O3")
 
#include <string.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
 
#include <math.h>
 ;
 
Tuple_Buffer_DeviceMemory createBuffer (VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties){
            VkBuffer buffer ;
    VkDeviceMemory bufferMemory ;
    {
                        VkBufferCreateInfo info  = {};
                info.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
                info.size=size;
                info.usage=usage;
                info.sharingMode=VK_SHARING_MODE_EXCLUSIVE;
                info.flags=0;
                        if ( !((VK_SUCCESS)==(vkCreateBuffer(state._device, &info, NULL, &buffer))) ) {
                                    {
                                                __auto_type current_time  = now();
                printf("%6.6f", ((current_time)-(state._start_time)));
                printf(" ");
                printf(printf_dec_format(__FILE__), __FILE__);
                printf(":");
                printf(printf_dec_format(__LINE__), __LINE__);
                printf(" ");
                printf(printf_dec_format(__func__), __func__);
                printf(" failed to (vkCreateBuffer (dot state _device) &info NULL &buffer): ");
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
            printf("  create buffer: ");
            printf(" buffer=");
            printf(printf_dec_format(buffer), buffer);
            printf(" (%s)", type_string(buffer));
            printf("\n");
};
};
        VkMemoryRequirements memReq ;
    vkGetBufferMemoryRequirements(state._device, buffer, &memReq);
    {
                        VkMemoryAllocateInfo info  = {};
                info.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
                info.allocationSize=memReq.size;
                info.memoryTypeIndex=findMemoryType(memReq.memoryTypeBits, properties);
                        if ( !((VK_SUCCESS)==(vkAllocateMemory(state._device, &info, NULL, &bufferMemory))) ) {
                                    {
                                                __auto_type current_time  = now();
                printf("%6.6f", ((current_time)-(state._start_time)));
                printf(" ");
                printf(printf_dec_format(__FILE__), __FILE__);
                printf(":");
                printf(printf_dec_format(__LINE__), __LINE__);
                printf(" ");
                printf(printf_dec_format(__func__), __func__);
                printf(" failed to (vkAllocateMemory (dot state _device) &info NULL &bufferMemory): ");
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
            printf(" bufferMemory=");
            printf(printf_dec_format(bufferMemory), bufferMemory);
            printf(" (%s)", type_string(bufferMemory));
            printf("\n");
};
};
    vkBindBufferMemory(state._device, buffer, bufferMemory, 0);
    return (Tuple_Buffer_DeviceMemory) {buffer, bufferMemory};
}
void generateMipmaps (VkImage image, VkFormat imageFormat, int32_t texWidth, int32_t texHeight, int32_t mipLevels){
        {
                        __auto_type current_time  = now();
        printf("%6.6f", ((current_time)-(state._start_time)));
        printf(" ");
        printf(printf_dec_format(__FILE__), __FILE__);
        printf(":");
        printf(printf_dec_format(__LINE__), __LINE__);
        printf(" ");
        printf(printf_dec_format(__func__), __func__);
        printf(" generateMipmaps: ");
        printf("\n");
};
            VkFormatProperties formatProperties ;
    vkGetPhysicalDeviceFormatProperties(state._physicalDevice, imageFormat, &formatProperties);
    if ( !(((formatProperties.optimalTilingFeatures) & (VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT))) ) {
                        {
                                    __auto_type current_time  = now();
            printf("%6.6f", ((current_time)-(state._start_time)));
            printf(" ");
            printf(printf_dec_format(__FILE__), __FILE__);
            printf(":");
            printf(printf_dec_format(__LINE__), __LINE__);
            printf(" ");
            printf(printf_dec_format(__func__), __func__);
            printf(" texture image format does not support linear blitting!: ");
            printf("\n");
};
};
            __auto_type commandBuffer  = beginSingleTimeCommands();
        VkImageMemoryBarrier barrier  = {};
        barrier.sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.image=image;
        barrier.srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;
        barrier.subresourceRange.aspectMask=VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.baseArrayLayer=0;
        barrier.subresourceRange.layerCount=1;
        barrier.subresourceRange.levelCount=1;
        __auto_type mipWidth  = texWidth;
    __auto_type mipHeight  = texHeight;
    for (int i=1;i<mipLevels;(i)++) {
                                barrier.subresourceRange.baseMipLevel=((i)-(1));
        barrier.oldLayout=VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout=VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask=VK_ACCESS_TRANSFER_READ_BIT;
        {
                                    __auto_type current_time  = now();
            printf("%6.6f", ((current_time)-(state._start_time)));
            printf(" ");
            printf(printf_dec_format(__FILE__), __FILE__);
            printf(":");
            printf(printf_dec_format(__LINE__), __LINE__);
            printf(" ");
            printf(printf_dec_format(__func__), __func__);
            printf("  vkCmdPipelineBarrier : ");
            printf(" i=");
            printf(printf_dec_format(i), i);
            printf(" (%s)", type_string(i));
            printf("\n");
};
        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &barrier);
                        __auto_type dstOffsetx  = 1;
        __auto_type dstOffsety  = 1;
        if ( 1<mipWidth ) {
                                                dstOffsetx=((mipWidth)/(2));
};
        if ( 1<mipHeight ) {
                                                dstOffsety=((mipHeight)/(2));
};
                VkImageBlit blit  = {};
                blit.srcOffsets[0]=(__typeof__(*blit.srcOffsets)) {0, 0, 0};
                blit.srcOffsets[1]=(__typeof__(*blit.srcOffsets)) {mipWidth, mipHeight, 1};
                blit.srcSubresource.aspectMask=VK_IMAGE_ASPECT_COLOR_BIT;
                blit.srcSubresource.mipLevel=((i)-(1));
                blit.srcSubresource.baseArrayLayer=0;
                blit.srcSubresource.layerCount=1;
                blit.dstOffsets[0]=(__typeof__(*blit.dstOffsets)) {0, 0, 0};
                blit.dstOffsets[1]=(__typeof__(*blit.dstOffsets)) {dstOffsetx, dstOffsety, 1};
                blit.dstSubresource.aspectMask=VK_IMAGE_ASPECT_COLOR_BIT;
                blit.dstSubresource.mipLevel=i;
                blit.dstSubresource.baseArrayLayer=0;
                blit.dstSubresource.layerCount=1;
        {
                                    __auto_type current_time  = now();
            printf("%6.6f", ((current_time)-(state._start_time)));
            printf(" ");
            printf(printf_dec_format(__FILE__), __FILE__);
            printf(":");
            printf(printf_dec_format(__LINE__), __LINE__);
            printf(" ");
            printf(printf_dec_format(__func__), __func__);
            printf("  vkCmdBlitImage: ");
            printf(" i=");
            printf(printf_dec_format(i), i);
            printf(" (%s)", type_string(i));
            printf("\n");
};
        vkCmdBlitImage(commandBuffer, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);
                                barrier.oldLayout=VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.newLayout=VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.srcAccessMask=VK_ACCESS_TRANSFER_READ_BIT;
        barrier.dstAccessMask=VK_ACCESS_SHADER_READ_BIT;
        {
                                    __auto_type current_time  = now();
            printf("%6.6f", ((current_time)-(state._start_time)));
            printf(" ");
            printf(printf_dec_format(__FILE__), __FILE__);
            printf(":");
            printf(printf_dec_format(__LINE__), __LINE__);
            printf(" ");
            printf(printf_dec_format(__func__), __func__);
            printf("  vkCmdPipelineBarrier: ");
            printf(" i=");
            printf(printf_dec_format(i), i);
            printf(" (%s)", type_string(i));
            printf("\n");
};
        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, NULL, 0, NULL, 1, &barrier);
                        if ( 1<mipWidth ) {
                                                mipWidth=((mipWidth)/(2));
};
        if ( 1<mipHeight ) {
                                                mipHeight=((mipHeight)/(2));
};
};
            barrier.subresourceRange.baseMipLevel=((state._mipLevels)-(1));
    barrier.oldLayout=VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout=VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask=VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, NULL, 0, NULL, 1, &barrier);
    endSingleTimeCommands(commandBuffer);
}
void copyBufferToImage (VkBuffer buffer, VkImage image, uint32_t width, uint32_t height){
            __auto_type commandBuffer  = beginSingleTimeCommands();
        VkBufferImageCopy region  = {};
        region.bufferOffset=0;
        region.bufferRowLength=0;
        region.bufferImageHeight=0;
        region.imageSubresource.aspectMask=VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel=0;
        region.imageSubresource.baseArrayLayer=0;
        region.imageSubresource.layerCount=1;
        region.imageOffset=(__typeof__(region.imageOffset)) {0, 0, 0};
        region.imageExtent=(__typeof__(region.imageExtent)) {width, height, 1};
    vkCmdCopyBufferToImage(commandBuffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    endSingleTimeCommands(commandBuffer);
}
void createTextureImage (){
        // uses command buffers 
        {
                        __auto_type current_time  = now();
        printf("%6.6f", ((current_time)-(state._start_time)));
        printf(" ");
        printf(printf_dec_format(__FILE__), __FILE__);
        printf(":");
        printf(printf_dec_format(__LINE__), __LINE__);
        printf(" ");
        printf(printf_dec_format(__func__), __func__);
        printf(" start loading texture: ");
        printf("\n");
};
            int texWidth  = 0;
    int texHeight  = 0;
    int texChannels  = 0;
    __auto_type texFilename  = "chalet.jpg";
    __auto_type pixels  = stbi_load(texFilename, &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
    VkDeviceSize imageSize  = ((texWidth)*(texHeight)*(4));
    if ( !(pixels) ) {
                        {
                                    __auto_type current_time  = now();
            printf("%6.6f", ((current_time)-(state._start_time)));
            printf(" ");
            printf(printf_dec_format(__FILE__), __FILE__);
            printf(":");
            printf(printf_dec_format(__LINE__), __LINE__);
            printf(" ");
            printf(printf_dec_format(__func__), __func__);
            printf(" failed to load texture image.: ");
            printf(" texFilename=");
            printf(printf_dec_format(texFilename), texFilename);
            printf(" (%s)", type_string(texFilename));
            printf("\n");
};
};
        state._mipLevels=(uint32_t) ((1)+(floor(log2(max(texWidth, texHeight)))));
    {
                        __auto_type current_time  = now();
        printf("%6.6f", ((current_time)-(state._start_time)));
        printf(" ");
        printf(printf_dec_format(__FILE__), __FILE__);
        printf(":");
        printf(printf_dec_format(__LINE__), __LINE__);
        printf(" ");
        printf(printf_dec_format(__func__), __func__);
        printf(" loaded texture: ");
        printf(" texWidth=");
        printf(printf_dec_format(texWidth), texWidth);
        printf(" (%s)", type_string(texWidth));
        printf(" texHeight=");
        printf(printf_dec_format(texHeight), texHeight);
        printf(" (%s)", type_string(texHeight));
        printf(" texChannels=");
        printf(printf_dec_format(texChannels), texChannels);
        printf(" (%s)", type_string(texChannels));
        printf(" texFilename=");
        printf(printf_dec_format(texFilename), texFilename);
        printf(" (%s)", type_string(texFilename));
        printf(" state._mipLevels=");
        printf(printf_dec_format(state._mipLevels), state._mipLevels);
        printf(" (%s)", type_string(state._mipLevels));
        printf("\n");
};
        // width    mipLevels
    // 2        2       
    // 4        3       
    // 16       5       
    // 32       6       
    // 128      8       
    // 255      8       
    // 256      9       
    // 257      9       
    // 512      10      
    // 1024     11      ;
        __auto_type stagingBufferTuple  = createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, ((VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) | (VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)));
    void* data  = NULL;
    {
                        __auto_type current_time  = now();
        printf("%6.6f", ((current_time)-(state._start_time)));
        printf(" ");
        printf(printf_dec_format(__FILE__), __FILE__);
        printf(":");
        printf(printf_dec_format(__LINE__), __LINE__);
        printf(" ");
        printf(printf_dec_format(__func__), __func__);
        printf(" map staging: ");
        printf("\n");
};
    vkMapMemory(state._device, stagingBufferTuple.memory, 0, imageSize, 0, &data);
    {
                        __auto_type current_time  = now();
        printf("%6.6f", ((current_time)-(state._start_time)));
        printf(" ");
        printf(printf_dec_format(__FILE__), __FILE__);
        printf(":");
        printf(printf_dec_format(__LINE__), __LINE__);
        printf(" ");
        printf(printf_dec_format(__func__), __func__);
        printf(" copy pixels: ");
        printf("\n");
};
    memcpy(data, pixels, imageSize);
    {
                        __auto_type current_time  = now();
        printf("%6.6f", ((current_time)-(state._start_time)));
        printf(" ");
        printf(printf_dec_format(__FILE__), __FILE__);
        printf(":");
        printf(printf_dec_format(__LINE__), __LINE__);
        printf(" ");
        printf(printf_dec_format(__func__), __func__);
        printf(" unmap staging: ");
        printf("\n");
};
    vkUnmapMemory(state._device, stagingBufferTuple.memory);
    stbi_image_free(pixels);
    {
                        __auto_type current_time  = now();
        printf("%6.6f", ((current_time)-(state._start_time)));
        printf(" ");
        printf(printf_dec_format(__FILE__), __FILE__);
        printf(":");
        printf(printf_dec_format(__LINE__), __LINE__);
        printf(" ");
        printf(printf_dec_format(__func__), __func__);
        printf(" create image: ");
        printf("\n");
};
        __auto_type imageTuple  = createImage(texWidth, texHeight, state._mipLevels, VK_SAMPLE_COUNT_1_BIT, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_TILING_OPTIMAL, ((VK_IMAGE_USAGE_TRANSFER_DST_BIT) | (VK_IMAGE_USAGE_TRANSFER_SRC_BIT) | (VK_IMAGE_USAGE_SAMPLED_BIT)), VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        state._textureImage=imageTuple.image;
    state._textureImageMemory=imageTuple.memory;
    {
                        __auto_type current_time  = now();
        printf("%6.6f", ((current_time)-(state._start_time)));
        printf(" ");
        printf(printf_dec_format(__FILE__), __FILE__);
        printf(":");
        printf(printf_dec_format(__LINE__), __LINE__);
        printf(" ");
        printf(printf_dec_format(__func__), __func__);
        printf(" transition image layout: ");
        printf("\n");
};
    transitionImageLayout(state._textureImage, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, state._mipLevels);
    copyBufferToImage(stagingBufferTuple.buffer, state._textureImage, texWidth, texHeight);
    {
                        __auto_type current_time  = now();
        printf("%6.6f", ((current_time)-(state._start_time)));
        printf(" ");
        printf(printf_dec_format(__FILE__), __FILE__);
        printf(":");
        printf(printf_dec_format(__LINE__), __LINE__);
        printf(" ");
        printf(printf_dec_format(__func__), __func__);
        printf(" destroy staging: ");
        printf("\n");
};
        vkDestroyBuffer(state._device, stagingBufferTuple.buffer, NULL);
    vkFreeMemory(state._device, stagingBufferTuple.memory, NULL);
    {
                        __auto_type current_time  = now();
        printf("%6.6f", ((current_time)-(state._start_time)));
        printf(" ");
        printf(printf_dec_format(__FILE__), __FILE__);
        printf(":");
        printf(printf_dec_format(__LINE__), __LINE__);
        printf(" ");
        printf(printf_dec_format(__func__), __func__);
        printf(" start mip maps: ");
        printf("\n");
};
    generateMipmaps(state._textureImage, VK_FORMAT_R8G8B8A8_UNORM, texWidth, texHeight, state._mipLevels);
    {
                        __auto_type current_time  = now();
        printf("%6.6f", ((current_time)-(state._start_time)));
        printf(" ");
        printf(printf_dec_format(__FILE__), __FILE__);
        printf(":");
        printf(printf_dec_format(__LINE__), __LINE__);
        printf(" ");
        printf(printf_dec_format(__func__), __func__);
        printf(" finished mip maps: ");
        printf("\n");
};
};