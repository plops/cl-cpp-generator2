 
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
 
#include "utils.h"
 
#include "globals.h"
 
#include "proto2.h"
 ;
extern State state;
void createTextureSampler (){
        {
                        VkSamplerCreateInfo info  = {};
                info.sType=VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
                info.magFilter=VK_FILTER_LINEAR;
                info.minFilter=VK_FILTER_LINEAR;
                info.addressModeU=VK_SAMPLER_ADDRESS_MODE_REPEAT;
                info.addressModeV=VK_SAMPLER_ADDRESS_MODE_REPEAT;
                info.addressModeW=VK_SAMPLER_ADDRESS_MODE_REPEAT;
                info.anisotropyEnable=VK_TRUE;
                info.maxAnisotropy=16;
                info.borderColor=VK_BORDER_COLOR_INT_OPAQUE_BLACK;
                info.unnormalizedCoordinates=VK_FALSE;
                info.compareEnable=VK_FALSE;
                info.compareOp=VK_COMPARE_OP_ALWAYS;
                info.mipmapMode=VK_SAMPLER_MIPMAP_MODE_LINEAR;
                info.mipLodBias=(0.0e+0f);
                info.minLod=(0.0e+0f);
                info.maxLod=(float) state._mipLevels;
                        if ( !((VK_SUCCESS)==(vkCreateSampler(state._device, &info, NULL, &(state._textureSampler)))) ) {
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
                printf(" failed to (vkCreateSampler (dot state _device) &info NULL            (ref (dot state _textureSampler))): ");
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
            printf("  create sampler: ");
            printf(" state._textureSampler=");
            printf(printf_dec_format(state._textureSampler), state._textureSampler);
            printf(" (%s)", type_string(state._textureSampler));
            printf("\n");
};
};
};