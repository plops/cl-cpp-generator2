 
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
                                                __auto_type current_time  = now();
                printf("%6.6f", ((current_time)-(state._start_time)));
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
                                    __auto_type current_time  = now();
            printf("%6.6f", ((current_time)-(state._start_time)));
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