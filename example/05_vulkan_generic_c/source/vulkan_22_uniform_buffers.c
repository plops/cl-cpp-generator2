 
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
    for (int i = 0;i<n;(i)+=(1)) {
                        __auto_type uniformBuffer  = createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, ((VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) | (VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)));
                state._uniformBuffers[i]=uniformBuffer.buffer;
        state._uniformBuffersMemory[i]=uniformBuffer.memory;
};
};