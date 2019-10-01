 
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
 
#include "utils.h"
 
#include "globals.h"
 
#include "proto2.h"
 ;
extern State state;
void createCommandPool (){
            __auto_type queueFamilyIndices  = findQueueFamilies(state._physicalDevice);
    {
                        VkCommandPoolCreateInfo info  = {};
                info.sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
                info.queueFamilyIndex=queueFamilyIndices.graphicsFamily;
                info.flags=0;
                        if ( !((VK_SUCCESS)==(vkCreateCommandPool(state._device, &info, NULL, &(state._commandPool)))) ) {
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
                printf(" failed to (vkCreateCommandPool (dot state _device) &info NULL            (ref (dot state _commandPool))): ");
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
            printf("  create command-pool: ");
            printf(" state._commandPool=");
            printf(printf_dec_format(state._commandPool), state._commandPool);
            printf(" (%s)", type_string(state._commandPool));
            printf("\n");
};
};
};