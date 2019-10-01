 
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
 
#include "utils.h"
 
#include "globals.h"
 
#include "proto2.h"
 ;
extern State state;
void createSyncObjects (){
            VkSemaphoreCreateInfo semaphoreInfo  = {};
        semaphoreInfo.sType=VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
            VkFenceCreateInfo fenceInfo  = {};
        fenceInfo.sType=VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags=VK_FENCE_CREATE_SIGNALED_BIT;
        for (int i = 0;i<_MAX_FRAMES_IN_FLIGHT;(i)+=(1)) {
                if ( !((VK_SUCCESS)==(vkCreateSemaphore(state._device, &semaphoreInfo, NULL, &(state._imageAvailableSemaphores[i])))) ) {
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
                printf(" failed to (vkCreateSemaphore (dot state _device) &semaphoreInfo NULL            (ref (aref (dot state _imageAvailableSemaphores) i))): ");
                printf("\n");
};
};
                if ( !((VK_SUCCESS)==(vkCreateSemaphore(state._device, &semaphoreInfo, NULL, &(state._renderFinishedSemaphores[i])))) ) {
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
                printf(" failed to (vkCreateSemaphore (dot state _device) &semaphoreInfo NULL            (ref (aref (dot state _renderFinishedSemaphores) i))): ");
                printf("\n");
};
};
                if ( !((VK_SUCCESS)==(vkCreateFence(state._device, &fenceInfo, NULL, &(state._inFlightFences[i])))) ) {
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
                printf(" failed to (vkCreateFence (dot state _device) &fenceInfo NULL            (ref (aref (dot state _inFlightFences) i))): ");
                printf("\n");
};
};
}
};