 
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
 
#include "utils.h"
 
#include "globals.h"
 
#include "proto2.h"
 ;
extern State state;
 
void createUniformBuffers (){
            __auto_type bufferSize  = sizeof(UniformBufferObject);
    __auto_type n  = length(state._swapChainImages);
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
        printf(" create uniform buffers: ");
        printf(" bufferSize=");
        printf(printf_dec_format(bufferSize), bufferSize);
        printf(" (%s)", type_string(bufferSize));
        printf(" length(state._swapChainImages)=");
        printf(printf_dec_format(length(state._swapChainImages)), length(state._swapChainImages));
        printf(" (%s)", type_string(length(state._swapChainImages)));
        printf(" length(state._uniformBuffers)=");
        printf(printf_dec_format(length(state._uniformBuffers)), length(state._uniformBuffers));
        printf(" (%s)", type_string(length(state._uniformBuffers)));
        printf(" length(state._uniformBuffersMemory)=");
        printf(printf_dec_format(length(state._uniformBuffersMemory)), length(state._uniformBuffersMemory));
        printf(" (%s)", type_string(length(state._uniformBuffersMemory)));
        printf("\n");
};
    for (int i = 0;i<n;(i)+=(1)) {
                        __auto_type uniformBuffer  = createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, ((VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) | (VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)));
                state._uniformBuffers[i]=uniformBuffer.buffer;
        state._uniformBuffersMemory[i]=uniformBuffer.memory;
};
};