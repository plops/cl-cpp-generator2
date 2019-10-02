 
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
 
#include "utils.h"
 
#include "globals.h"
 
#include "proto2.h"
 ;
extern State state;
void createDescriptorSetLayout (){
            VkDescriptorSetLayoutBinding samplerLayoutBinding  = {};
        samplerLayoutBinding.binding=1;
        samplerLayoutBinding.descriptorCount=1;
        samplerLayoutBinding.descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        samplerLayoutBinding.pImmutableSamplers=NULL;
        samplerLayoutBinding.stageFlags=VK_SHADER_STAGE_FRAGMENT_BIT;
            VkDescriptorSetLayoutBinding uboLayoutBinding  = {};
        uboLayoutBinding.binding=0;
        uboLayoutBinding.descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        uboLayoutBinding.descriptorCount=1;
        uboLayoutBinding.stageFlags=VK_SHADER_STAGE_VERTEX_BIT;
        uboLayoutBinding.pImmutableSamplers=NULL;
            VkDescriptorSetLayoutBinding bindings[]  = {uboLayoutBinding, samplerLayoutBinding};
    {
                        VkDescriptorSetLayoutCreateInfo info  = {};
                info.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
                info.bindingCount=length(bindings);
                info.pBindings=bindings;
                        if ( !((VK_SUCCESS)==(vkCreateDescriptorSetLayout(state._device, &info, NULL, &(state._descriptorSetLayout)))) ) {
                                    {
                                                __auto_type current_time  = now();
                printf("%6.6f", ((current_time)-(state._start_time)));
                printf(" ");
                printf(printf_dec_format(__FILE__), __FILE__);
                printf(":");
                printf(printf_dec_format(__LINE__), __LINE__);
                printf(" ");
                printf(printf_dec_format(__func__), __func__);
                printf(" failed to (vkCreateDescriptorSetLayout (dot state _device) &info NULL            (ref (dot state _descriptorSetLayout))): ");
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
            printf("  create descriptor-set-layout: ");
            printf(" state._descriptorSetLayout=");
            printf(printf_dec_format(state._descriptorSetLayout), state._descriptorSetLayout);
            printf(" (%s)", type_string(state._descriptorSetLayout));
            printf("\n");
};
};
};