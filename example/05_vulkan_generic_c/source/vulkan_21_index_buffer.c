 
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
 
#include "utils.h"
 
#include "globals.h"
 
#include "proto2.h"
 ;
extern State state;
#include <string.h>
void createIndexBuffer (){
            __auto_type bufferSize  = ((sizeof(state._indices[0]))*(state._num_indices));
    __auto_type stagingBuffer  = createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, ((VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) | (VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)));
        void* data ;
    vkMapMemory(state._device, stagingBuffer.memory, 0, bufferSize, 0, &data);
    memcpy(data, state._indices, bufferSize);
    vkUnmapMemory(state._device, stagingBuffer.memory);
            __auto_type indexBuffer  = createBuffer(bufferSize, ((VK_BUFFER_USAGE_INDEX_BUFFER_BIT) | (VK_BUFFER_USAGE_TRANSFER_DST_BIT)), VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        state._indexBuffer=indexBuffer.buffer;
    state._indexBufferMemory=indexBuffer.memory;
    copyBuffer(stagingBuffer.buffer, state._indexBuffer, bufferSize);
            vkDestroyBuffer(state._device, stagingBuffer.buffer, NULL);
    vkFreeMemory(state._device, stagingBuffer.memory, NULL);
};