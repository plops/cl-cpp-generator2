 
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
 
#include "utils.h"
 
#include "globals.h"
 
#include "proto2.h"
 ;
extern State state;
void createDescriptorSets (){
            const int n  = length(state._swapChainImages);
    VkDescriptorSetLayout layouts[]  = {state._descriptorSetLayout, state._descriptorSetLayout, state._descriptorSetLayout, state._descriptorSetLayout};
    {
                        VkDescriptorSetAllocateInfo info  = {};
                info.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                info.descriptorPool=state._descriptorPool;
                info.descriptorSetCount=n;
                info.pSetLayouts=layouts;
                        if ( !((VK_SUCCESS)==(vkAllocateDescriptorSets(state._device, &info, state._descriptorSets))) ) {
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
                printf(" failed to (vkAllocateDescriptorSets (dot state _device) &info            (dot state _descriptorSets)): ");
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
            printf("  allocate descriptor-set: ");
            printf("\n");
};
};
    for (int i = 0;i<n;(i)+=(1)) {
                        VkDescriptorBufferInfo bufferInfo  = {};
                bufferInfo.buffer=state._uniformBuffers[i];
                bufferInfo.offset=0;
                bufferInfo.range=sizeof(UniformBufferObject);
                        VkWriteDescriptorSet uboDescriptorWrite  = {};
                uboDescriptorWrite.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                uboDescriptorWrite.dstSet=state._descriptorSets[i];
                uboDescriptorWrite.dstBinding=0;
                uboDescriptorWrite.dstArrayElement=0;
                uboDescriptorWrite.descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                uboDescriptorWrite.descriptorCount=1;
                uboDescriptorWrite.pBufferInfo=&bufferInfo;
                uboDescriptorWrite.pImageInfo=NULL;
                uboDescriptorWrite.pTexelBufferView=NULL;
                        VkDescriptorImageInfo imageInfo  = {};
                imageInfo.imageLayout=VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                imageInfo.imageView=state._textureImageView;
                imageInfo.sampler=state._textureSampler;
                        VkWriteDescriptorSet samplerDescriptorWrite  = {};
                samplerDescriptorWrite.sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                samplerDescriptorWrite.dstSet=state._descriptorSets[i];
                samplerDescriptorWrite.dstBinding=1;
                samplerDescriptorWrite.dstArrayElement=0;
                samplerDescriptorWrite.descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                samplerDescriptorWrite.descriptorCount=1;
                samplerDescriptorWrite.pBufferInfo=NULL;
                samplerDescriptorWrite.pImageInfo=&imageInfo;
                samplerDescriptorWrite.pTexelBufferView=NULL;
                        VkWriteDescriptorSet descriptorWrites[]  = {uboDescriptorWrite, samplerDescriptorWrite};
        vkUpdateDescriptorSets(state._device, length(descriptorWrites), descriptorWrites, 0, NULL);
};
};