 
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
 
#include "utils.h"
 
#include "globals.h"
 
#include "proto2.h"
 ;
extern State state;
void createDescriptorPool (){
            __auto_type n  = length(state._swapChainImages);
        VkDescriptorPoolSize uboPoolSize  = {};
        uboPoolSize.type=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        uboPoolSize.descriptorCount=n;
        VkDescriptorPoolSize samplerPoolSize  = {};
        samplerPoolSize.type=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        samplerPoolSize.descriptorCount=n;
        VkDescriptorPoolSize poolSizes[]  = {uboPoolSize, samplerPoolSize};
    {
                        VkDescriptorPoolCreateInfo info  = {};
                info.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
                info.poolSizeCount=length(poolSizes);
                info.pPoolSizes=poolSizes;
                info.maxSets=n;
                info.flags=0;
                        if ( !((VK_SUCCESS)==(vkCreateDescriptorPool(state._device, &info, NULL, &(state._descriptorPool)))) ) {
                                    {
                                                __auto_type current_time  = now();
                printf("%6.6f", ((current_time)-(state._start_time)));
                printf(" ");
                printf(printf_dec_format(__FILE__), __FILE__);
                printf(":");
                printf(printf_dec_format(__LINE__), __LINE__);
                printf(" ");
                printf(printf_dec_format(__func__), __func__);
                printf(" failed to (vkCreateDescriptorPool (dot state _device) &info NULL            (ref (dot state _descriptorPool))): ");
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
            printf("  create descriptor-pool: ");
            printf(" state._descriptorPool=");
            printf(printf_dec_format(state._descriptorPool), state._descriptorPool);
            printf(" (%s)", type_string(state._descriptorPool));
            printf("\n");
};
};
};