 
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
 
#include "utils.h"
 
#include "globals.h"
 
#include "proto2.h"
 ;
extern State state;
#include <string.h>
void copyBuffer (VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size){
            __auto_type commandBuffer  = beginSingleTimeCommands();
        VkBufferCopy copyRegion  = {};
        copyRegion.srcOffset=0;
        copyRegion.dstOffset=0;
        copyRegion.size=size;
    vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);
    endSingleTimeCommands(commandBuffer);
}
void createVertexBuffer (){
            __auto_type bufferSize  = ((sizeof(state._vertices[0]))*(state._num_vertices));
    __auto_type stagingBuffer  = createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, ((VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) | (VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)));
        void* data ;
    vkMapMemory(state._device, stagingBuffer.memory, 0, bufferSize, 0, &data);
    {
                        __auto_type current_time  = now();
        printf("%6.6f", ((current_time)-(state._start_time)));
        printf(" ");
        printf(printf_dec_format(__FILE__), __FILE__);
        printf(":");
        printf(printf_dec_format(__LINE__), __LINE__);
        printf(" ");
        printf(printf_dec_format(__func__), __func__);
        printf(" copy vertex buffer: ");
        printf(" bufferSize=");
        printf(printf_dec_format(bufferSize), bufferSize);
        printf(" (%s)", type_string(bufferSize));
        printf(" state._num_vertices=");
        printf(printf_dec_format(state._num_vertices), state._num_vertices);
        printf(" (%s)", type_string(state._num_vertices));
        printf("\n");
};
    memcpy(data, state._vertices, bufferSize);
    vkUnmapMemory(state._device, stagingBuffer.memory);
            __auto_type vertexBuffer  = createBuffer(bufferSize, ((VK_BUFFER_USAGE_VERTEX_BUFFER_BIT) | (VK_BUFFER_USAGE_TRANSFER_DST_BIT)), VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        state._vertexBuffer=vertexBuffer.buffer;
    state._vertexBufferMemory=vertexBuffer.memory;
    copyBuffer(stagingBuffer.buffer, state._vertexBuffer, bufferSize);
            vkDestroyBuffer(state._device, stagingBuffer.buffer, NULL);
    vkFreeMemory(state._device, stagingBuffer.memory, NULL);
};