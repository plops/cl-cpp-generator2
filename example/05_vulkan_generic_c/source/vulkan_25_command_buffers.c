 
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
 
#include "utils.h"
 
#include "globals.h"
 
#include "proto2.h"
 ;
extern State state;
void createCommandBuffers (){
        {
                        VkCommandBufferAllocateInfo info  = {};
                info.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
                info.commandPool=state._commandPool;
                info.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY;
                info.commandBufferCount=length(state._commandBuffers);
                        if ( !((VK_SUCCESS)==(vkAllocateCommandBuffers(state._device, &info, state._commandBuffers))) ) {
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
                printf(" failed to (vkAllocateCommandBuffers (dot state _device) &info            (dot state _commandBuffers)): ");
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
            printf("  allocate command-buffer: ");
            printf("\n");
};
};
        for (int i = 0;i<length(state._commandBuffers);(i)+=(1)) {
                {
                                    VkCommandBufferBeginInfo info  = {};
                        info.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
                        info.flags=0;
                        info.pInheritanceInfo=NULL;
                                    if ( !((VK_SUCCESS)==(vkBeginCommandBuffer(state._commandBuffers[i], &info))) ) {
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
                    printf(" failed to (vkBeginCommandBuffer (aref (dot state _commandBuffers) i) &info): ");
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
                printf("  begin command-buffer: ");
                printf(" state._commandBuffers[i]=");
                printf(printf_dec_format(state._commandBuffers[i]), state._commandBuffers[i]);
                printf(" (%s)", type_string(state._commandBuffers[i]));
                printf("\n");
};
};
                        VkClearValue clearColor  = {};
                clearColor.color=(__typeof__(clearColor.color)) {(0.0e+0f), (0.0e+0f), (0.0e+0f), (1.e+0f)};
                        VkClearValue clearDepth  = {};
                clearDepth.depthStencil=(__typeof__(clearDepth.depthStencil)) {(1.e+0f), 0};
                        VkClearValue clearValues[]  = {clearColor, clearDepth};
                VkRenderPassBeginInfo renderPassInfo  = {};
                renderPassInfo.sType=VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
                renderPassInfo.renderPass=state._renderPass;
                renderPassInfo.framebuffer=state._swapChainFramebuffers[i];
                renderPassInfo.renderArea.offset=(__typeof__(renderPassInfo.renderArea.offset)) {0, 0};
                renderPassInfo.renderArea.extent=state._swapChainExtent;
                renderPassInfo.clearValueCount=length(clearValues);
                renderPassInfo.pClearValues=clearValues;
                vkCmdBeginRenderPass(state._commandBuffers[i], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
                vkCmdBindPipeline(state._commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, state._graphicsPipeline);
                        VkBuffer vertexBuffers[]  = {state._vertexBuffer};
        VkDeviceSize offsets[]  = {0};
        vkCmdBindVertexBuffers(state._commandBuffers[i], 0, 1, vertexBuffers, offsets);
                vkCmdBindIndexBuffer(state._commandBuffers[i], state._indexBuffer, 0, VK_INDEX_TYPE_UINT32);
                vkCmdBindDescriptorSets(state._commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, state._pipelineLayout, 0, 1, &(state._descriptorSets[i]), 0, NULL);
                vkCmdDrawIndexed(state._commandBuffers[i], state._num_indices, 1, 0, 0, 0);
                vkCmdEndRenderPass(state._commandBuffers[i]);
                if ( !((VK_SUCCESS)==(vkEndCommandBuffer(state._commandBuffers[i]))) ) {
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
                printf(" failed to (vkEndCommandBuffer (aref (dot state _commandBuffers) i)): ");
                printf("\n");
};
};
}
};